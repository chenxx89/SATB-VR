import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CogVLM2_Captioner(torch.nn.Module):
    def __init__(self, model_path, torch_type=torch.bfloat16):
        super().__init__()
        self.torch_type = torch_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=self.torch_type, 
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            offload_buffers=True,
        ).eval()

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype
    
    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    
    def forward(
        self,
        video,
        prompt='Please describe this video in detail.',
        temperature=0.1, fps=None
    ):
        '''
        video: str or tensor
            str: the video path
            tensor: [T, C, H, W], torch.float32, (0,1)
        '''
        video = video * 255  # (0,1) -> (0,255)
        fps = fps if fps else min(15, video.shape[0])
        indices = self.get_index(video.shape[0], fps)
        video = video[indices]
        video = video.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
        response = self.predict(prompt, video, temperature)
        return str(response).strip()
    
    def predict(self, prompt, video, temperature):
        inputs = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=prompt,
            images=[video],
            history=[],
            template_version='chat'
        )
        device = self.device
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
            'images': [[inputs['images'][0].to(device=device, dtype=self.torch_type)]],
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
    
    def get_index(self, total_frames, fps):
        fps = round(fps)
        num_segments = max(total_frames // fps, 1)
        index_list = [i * fps for i in range(num_segments)]
        if index_list[-1] != total_frames - 1:
            index_list.append(total_frames - 1)
        return index_list
