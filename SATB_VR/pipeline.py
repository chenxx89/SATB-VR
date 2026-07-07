import math
import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX, DiffusionPipeline, logging
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from .acceleration import optimize_transformer
from .captioner import CogVLM2_Captioner
from .controlnet import CogVideoXControlnet
from .download_weights import download_ckpts
from .rope import prepare_rotary_positional_embeddings
from .scheduler import CogVideoXDPMImprovedScheduler
from .text_encoder import compute_prompt_embeddings
from .tiling import prepare_tiling_infos_generator
from .transformer import CogVideoXVRTransformer

class CogVideoXVRPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "captioner_model->text_encoder->predictor->controlnet->transformer->vae"

    def __init__(self,
        tokenizer,
        text_encoder,
        captioner_model,
        predictor,
        vae,
        transformer,
        controlnet,
        scheduler,
        ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            captioner_model=captioner_model,
            predictor=predictor,
            vae=vae,
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
        )

    @classmethod
    def from_args(cls, args):
        logging.set_verbosity_error()

        if torch.cuda.is_bf16_supported():
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float16

        download_ckpts(args.ckpt_path, args.enable_captioner)
        cogvideox_ckpt_path = os.path.join(args.ckpt_path, "CogVideoX1.5-5B")
        cogvlm2_ckpt_path = os.path.join(args.ckpt_path, "cogvlm2-llama3-caption")
        satb_ckpt_path = os.path.join(args.ckpt_path, "SATB-VR")

        # Text encoders
        if args.enable_text_encoder:
            print("Loading tokenizer and text encoder")
            tokenizer = AutoTokenizer.from_pretrained(
                cogvideox_ckpt_path, subfolder="tokenizer")
            text_encoder = T5EncoderModel.from_pretrained(
                cogvideox_ckpt_path, subfolder="text_encoder")
            text_encoder.requires_grad_(False)
            text_encoder.to(dtype=weight_dtype)
            if args.enable_captioner:
                print("Loading CogVLM2 captioner")
                captioner_model = CogVLM2_Captioner(model_path=cogvlm2_ckpt_path, torch_type=weight_dtype)
            else:
                captioner_model = None
            prompt_embeds = negative_prompt_embeds = None
        else:
            print("Use precomputed text embeddings")
            tokenizer = None
            text_encoder = None
            captioner_model = None
            prompt_embeds = torch.load(os.path.join(satb_ckpt_path, "prompt_embeds.pt"))
            negative_prompt_embeds = torch.load(os.path.join(satb_ckpt_path, "negative_prompt_embeds.pt"))

        # Scheduler
        scheduler = CogVideoXDPMImprovedScheduler.from_pretrained(cogvideox_ckpt_path, subfolder="scheduler")

        # Transformer
        print("Loading transformer")
        transformer = CogVideoXVRTransformer.from_pretrained(
            cogvideox_ckpt_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=False,
            _class_name='CogVideoXVRTransformer',
            enable_connector=True,
            enable_control_patchemb=True,
        )
        for module in ["connectors", "control_patch_embed"]:
            state_dict = torch.load(
                os.path.join(satb_ckpt_path, f"{module}.pt"),
                map_location='cpu'
            )
            getattr(transformer, module).load_state_dict(state_dict)
        transformer.load_lora_adapter(os.path.join(satb_ckpt_path, "lora_transformer"),
            use_safetensors=True, adapter_name='default', prefix=None)
        transformer.requires_grad_(False)
        transformer.to(dtype=weight_dtype)

        predictor, loading_info = CogVideoXVRTransformer.from_pretrained(
            cogvideox_ckpt_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            output_loading_info=True,
            _class_name='CogVideoXVRTransformer',
        )
        predictor.load_lora_adapter(os.path.join(satb_ckpt_path, "lora_predictor"),
            use_safetensors=True, adapter_name='default', prefix=None)
        predictor.requires_grad_(False)
        predictor.to(dtype=weight_dtype)
        print(f"Loading predictor done, loading info: {loading_info}")

        # VAE
        print("Loading vae")
        vae = AutoencoderKLCogVideoX.from_pretrained(cogvideox_ckpt_path, subfolder="vae")
        vae.requires_grad_(False)
        vae.to(dtype=weight_dtype)
        vae.enable_slicing()
        vae.enable_tiling()

        # ControlNet
        controlnet = CogVideoXControlnet.from_pretrained(
            pretrained_model_name_or_path=satb_ckpt_path,
            subfolder="controlnet")
        controlnet.load_lora_adapter(os.path.join(satb_ckpt_path, "lora_controlnet"),
            use_safetensors=True, adapter_name='default', prefix=None)
        controlnet.requires_grad_(False)
        controlnet.to(dtype=weight_dtype)

        # Acceleration optimizations
        transformer = optimize_transformer(transformer)
        controlnet = optimize_transformer(controlnet)

        print("***** Loading Model Done *****")

        logging.set_verbosity_warning()

        pipe = cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            captioner_model=captioner_model,
            predictor=predictor,
            vae=vae,
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        
        pipe.args = args
        pipe.weight_dtype = weight_dtype
        pipe.tile_size = 1024
        pipe.tile_stride = 768
        pipe.temporal_tile_size = 80
        pipe.temporal_tile_stride = 64
        pipe.max_sequence_length = 226
        pipe.prompt_embeds = prompt_embeds
        pipe.negative_prompt_embeds = negative_prompt_embeds
        
        return pipe
    
    def get_inference_timesteps(self, num_inference_steps):
        if num_inference_steps <= 2:
            base_max_time = 200
        elif num_inference_steps == 3:
            base_max_time = 300
        else:
            base_max_time = 400
            
        stride = base_max_time // num_inference_steps
        
        timesteps = [(i * stride) - 1 for i in range(num_inference_steps, 0, -1)]

        return timesteps
    
    def prepare_prompts(self, input_video, fps=24):
        POS_PROMPT = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
        NEG_PROMPT = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth.'

        input_video = torch.cat([input_video[:1].repeat(7, 1, 1, 1), input_video], dim=0)
        tiling_infos = list(prepare_tiling_infos_generator(
            self.args.enable_spatial_tiling,
            self.args.enable_temporal_tiling,
            input_video.unsqueeze(0), self.tile_size, self.tile_stride,
            self.temporal_tile_size, self.temporal_tile_stride,
        ))
        print(f"Captioning video with {len(tiling_infos)} tiles")
        input_video = input_video.to(self._execution_device)
        prompts = []
        for (tile_slice, _) in tiling_infos:
            tile_video = input_video[tile_slice[1:]]
            if self.captioner_model is not None:
                with torch.no_grad():
                    response = self.captioner_model(tile_video, fps=fps)
                    prompts.append(response)
            else:
                prompts.append('')

        prompt_list = [f"{prompt} {POS_PROMPT}" for prompt in prompts]
        negative_prompt_list = [NEG_PROMPT for _ in range(len(prompts))]
        return prompt_list, negative_prompt_list


    @torch.no_grad()
    def __call__(
        self,
        args,
        input_video,
        fps=24,
        generator=None,
    ):
        self.args = args
        device = self._execution_device
        model_config = self.transformer.config

        ori_height, ori_width = input_video.shape[-2:]

        # Resize input_video
        height = 8 * math.ceil(ori_height / 8)
        width = 8 * math.ceil(ori_width / 8)
        input_video = F.interpolate(input_video, size=(height, width), mode='bicubic')

        # Pad input_video
        num_padding_frames = 0
        if (input_video.size(0) - 1) % 8 != 0:
            num_padding_frames = 8 - (input_video.size(0) - 1) % 8
            input_video = torch.cat([input_video, input_video[-1:].repeat(num_padding_frames, 1, 1, 1)], dim=0)

        batch_size = 1
        do_classifier_free_guidance = args.guidance_scale > 1.0

        # Encode prompt
        if self.text_encoder is not None:
            prompt, negative_prompt = self.prepare_prompts(input_video, fps)
            prompt_embeds = compute_prompt_embeddings(
                self.tokenizer, self.text_encoder, prompt,
                self.max_sequence_length, device, self.weight_dtype,
                requires_grad=False, offload_model=False,
            )
            if do_classifier_free_guidance:
                negative_prompt_embeds = compute_prompt_embeddings(
                    self.tokenizer, self.text_encoder, negative_prompt,
                    self.max_sequence_length, device, self.weight_dtype,
                    requires_grad=False, offload_model=False,
                )
            else:
                negative_prompt_embeds = None
        else:
            prompt_embeds = self.prompt_embeds
            negative_prompt_embeds = self.negative_prompt_embeds

        prompt_embeds = prompt_embeds.to(device, dtype=self.weight_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device, dtype=self.weight_dtype)

        # VAE encode [B, C, F, H, W]
        input_video = input_video.unsqueeze(0).to(device=device, dtype=self.weight_dtype)
        input_video = input_video.permute(0, 2, 1, 3, 4)
        input_latents = self.vae.encode(input_video.mul(2.0).sub(1.0)).latent_dist.sample(generator)
        input_latents = input_latents * self.vae.config.scaling_factor
        input_latents = input_latents.permute(0, 2, 1, 3, 4).to(dtype=self.weight_dtype)

        # Pad for patch_size_t
        num_latent_padding_frames = 0
        if model_config.patch_size_t is not None and input_latents.shape[1] % model_config.patch_size_t > 0:
            ncopy = input_latents.shape[1] % model_config.patch_size_t
            first_frame = input_latents[:, :ncopy, ...]
            input_latents = torch.cat([first_frame, input_latents], dim=1)
            num_latent_padding_frames = ncopy

        # Prepare noise steps
        timesteps = self.get_inference_timesteps(args.num_inference_steps)
        noise = torch.randn_like(input_latents)

        timesteps = [torch.full((batch_size,), fill_value=timesteps[i], dtype=torch.int64, 
                                device=device) for i in range(args.num_inference_steps)]

        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device)
        self.scheduler.num_inference_steps = args.num_inference_steps

        # Tiling
        tiling_infos = list(prepare_tiling_infos_generator(
            enable_spatial_tiling=args.enable_spatial_tiling,
            enable_temporal_tiling=args.enable_temporal_tiling,
            latents=input_latents,
            tile_size=self.tile_size // 8, 
            tile_stride=self.tile_stride // 8,
            temporal_tile_size=self.temporal_tile_size // 4, 
            temporal_tile_stride=self.temporal_tile_stride // 4,
        ))

        old_pred_original_sample = None

        latents_meshgrid = torch.zeros_like(input_latents)
        weights_meshgrid = torch.zeros_like(input_latents)

        for tile_index, (tile_slice, tile_weights) in enumerate(tiling_infos):
            tile_input_latents = input_latents[tile_slice]
            prompt_slice = slice(tile_index, tile_index + 1) if self.text_encoder is not None else slice(None)
            tile_prompt_embeds = prompt_embeds[prompt_slice]

            image_rotary_emb = (
                prepare_rotary_positional_embeddings(
                    latent_height=tile_input_latents.shape[-2], latent_width=tile_input_latents.shape[-1],
                    num_frames=tile_input_latents.shape[1], patch_size=model_config.patch_size,
                    patch_size_t=model_config.patch_size_t, attention_head_dim=model_config.attention_head_dim,
                    device=device, sample_height=model_config.sample_height, sample_width=model_config.sample_width,
                ) if model_config.use_rotary_positional_embeddings else None
            )

            predictor_t = torch.full((batch_size,), fill_value=399, dtype=torch.int64, device=device)
            tile_predicted = self.predictor(
                hidden_states=tile_input_latents, encoder_hidden_states=tile_prompt_embeds,
                timestep=predictor_t, image_rotary_emb=image_rotary_emb,
                return_dict=False
            )[0]
            tile_predicted = self.scheduler.get_velocity(tile_predicted, tile_input_latents, predictor_t)
            latents_meshgrid[tile_slice] += tile_predicted * tile_weights
            weights_meshgrid[tile_slice] += tile_weights

        latents = latents_meshgrid / weights_meshgrid
        latents = self.scheduler.add_noise(latents, noise, timesteps[0]).to(dtype=self.weight_dtype)

        
        for index, timestep in enumerate(tqdm(timesteps, desc="Inference")):
            do_cfg = do_classifier_free_guidance

            latents_meshgrid = torch.zeros_like(latents)
            old_pred_meshgrid = torch.zeros_like(latents)
            weights_meshgrid = torch.zeros_like(latents)

            for tile_index, (tile_slice, tile_weights) in enumerate(tiling_infos):
                tile_latents = latents[tile_slice]
                tile_input_latents = input_latents[tile_slice]
                prompt_slice = slice(tile_index, tile_index + 1) if self.text_encoder is not None else slice(None)
                tile_prompt_embeds = prompt_embeds[prompt_slice]
                tile_neg_embeds = negative_prompt_embeds[prompt_slice] if do_cfg and negative_prompt_embeds is not None else None
                tile_old_pred = old_pred_original_sample[tile_slice] if old_pred_original_sample is not None else None

                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        latent_height=tile_latents.shape[-2], latent_width=tile_latents.shape[-1],
                        num_frames=tile_latents.shape[1], patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t, attention_head_dim=model_config.attention_head_dim,
                        device=device, sample_height=model_config.sample_height, sample_width=model_config.sample_width,
                    ) if model_config.use_rotary_positional_embeddings else None
                )

                latent_model_input = torch.cat([tile_latents] * 2) if do_cfg else tile_latents
                input_latents_expand = torch.cat([tile_input_latents] * 2) if do_cfg else tile_input_latents
                t_expand = timestep.expand(latent_model_input.shape[0])
                combined_embeds = torch.cat([tile_neg_embeds, tile_prompt_embeds], dim=0) if do_cfg else tile_prompt_embeds

                # ControlNet
                ctrl_states = self.controlnet(
                    hidden_states=latent_model_input, encoder_hidden_states=combined_embeds,
                    control_states=input_latents_expand, image_rotary_emb=image_rotary_emb,
                    timestep=t_expand, return_dict=False,
                )[0]
                ctrl_states = [s.to(prompt_embeds.dtype) for s in ctrl_states]

                # Transformer
                tile_model_input = torch.cat([latent_model_input, input_latents_expand], dim=2)
                tile_pred = self.transformer(
                    hidden_states=tile_model_input, encoder_hidden_states=combined_embeds,
                    control_hidden_states=ctrl_states, image_rotary_emb=image_rotary_emb,
                    timestep=t_expand, return_dict=False
                )[0].float()

                # CFG
                if do_cfg:
                    uncond, text = tile_pred.chunk(2)
                    tile_pred = uncond + args.guidance_scale * (text - uncond)

                # step
                extra_kwargs = {'prev_timestep': timesteps[index + 1] if index < len(timesteps) - 1 else -1}
                tile_pred, tile_old_pred = self.scheduler.step(
                    tile_pred,
                    tile_old_pred if index > 0 else None,
                    timestep, timesteps[index - 1] if index > 0 else None,
                    sample=tile_latents, return_dict=False,
                    **extra_kwargs,
                )

                latents_meshgrid[tile_slice] += tile_pred * tile_weights
                old_pred_meshgrid[tile_slice] += tile_old_pred * tile_weights
                weights_meshgrid[tile_slice] += tile_weights

            latents = latents_meshgrid / weights_meshgrid
            old_pred_original_sample = old_pred_meshgrid / weights_meshgrid

        # Decode
        if num_latent_padding_frames > 0:
            latents = latents[:, num_latent_padding_frames:]
        latents_for_decode = latents.permute(0, 2, 1, 3, 4)
        latents_for_decode = 1 / self.vae.config.scaling_factor * latents_for_decode
        video = self.vae.decode(latents_for_decode).sample
        video = video.mul(0.5).add(0.5).clip(0, 1)

        # Resize back to original resolution
        video = video.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        video = [F.interpolate(v, size=(ori_height, ori_width), mode='bilinear') for v in video]
        video = torch.stack(video, dim=0)
        if num_padding_frames > 0:
            video = video[:, :-num_padding_frames, :, :, :]
        return video

