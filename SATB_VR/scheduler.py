import torch
from typing import Optional, Tuple, Union
from diffusers.schedulers.scheduling_dpm_cogvideox import (
    CogVideoXDPMScheduler, DDIMSchedulerOutput, randn_tensor
)


class CogVideoXDPMImprovedScheduler(CogVideoXDPMScheduler):
    def step(
        self,
        model_output: torch.Tensor,
        old_pred_original_sample: torch.Tensor,
        timestep: int,
        timestep_back: int,
        prev_timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: torch.Generator | None = None,
        variance_noise: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> DDIMSchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            old_pred_original_sample (`torch.Tensor`):
                The predicted original sample from the previous timestep.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            timestep_back (`int`):
                The timestep to look back to.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_back = self.alphas_cumprod[timestep_back] if timestep_back is not None else None

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        h, r, lamb, lamb_next = self.get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
        mult = list(self.get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back))
        mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

        noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
        prev_sample = mult[0] * sample - mult[1] * pred_original_sample + mult_noise * noise

        if old_pred_original_sample is None or prev_timestep < 0:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return prev_sample, pred_original_sample
        else:
            denoised_d = mult[2] * pred_original_sample - mult[3] * old_pred_original_sample
            noise = randn_tensor(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            x_advanced = mult[0] * sample - mult[1] * denoised_d + mult_noise * noise

            prev_sample = x_advanced

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
