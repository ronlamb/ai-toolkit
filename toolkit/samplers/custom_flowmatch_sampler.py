import math
from typing import Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np
from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0
        self.timestep_type = "linear"

        with torch.no_grad():
            self.default_weighing_tensor = torch.tensor(
               default_weighing_scheme, dtype=torch.float32
)
            # create weights for timesteps
            num_timesteps = 1000
            # Bell-Shaped Mean-Normalized Timestep Weighting
            # bsmntw? need a better name

            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

            # Shift minimum to 0
            y_shifted = y - y.min()

            # Scale to make mean 1
            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # only do half bell
            hbsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # flatten second half to max
            hbsmntw_weighing[num_timesteps //
                             2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            # Create linear timesteps from 1000 to 1
            timesteps = torch.linspace(1000, 1, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            
            # Cache for device-aware weight tensors to prevent repeated .to() calls
            self._cached_device = None
            self._cached_dtype = None
            self._default_weighing_tensor_cached = None
            self._linear_timesteps_weights_cached = None
            self._linear_timesteps_weights2_cached = None
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False, timestep_type="linear") -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = self._get_step_indices(timesteps.to(self.timesteps.device))

        # Invalidate cache if device or dtype changed
        if self._cached_device != timesteps.device or self._cached_dtype != timesteps.dtype:
            self._cached_device = timesteps.device
            self._cached_dtype = timesteps.dtype
            # Create cached copies on target device
            self._default_weighing_tensor_cached = self.default_weighing_tensor.to(device=timesteps.device, dtype=timesteps.dtype)
            self._linear_timesteps_weights_cached = self.linear_timesteps_weights.to(device=timesteps.device, dtype=timesteps.dtype)
            self._linear_timesteps_weights2_cached = self.linear_timesteps_weights2.to(device=timesteps.device, dtype=timesteps.dtype)

        # Get the weights for the timesteps (use cached tensors)
        if timestep_type == "weighted":
            weights = self._default_weighing_tensor_cached[step_indices]
        elif v2:
            weights = self._linear_timesteps_weights2_cached[step_indices]
        else:
            weights = self._linear_timesteps_weights_cached[step_indices]
        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        step_indices = self._get_step_indices(timesteps.to(self.timesteps.device))
        sigmas = self.sigmas[step_indices].to(device=device, dtype=dtype)

        while len(sigmas.shape) < n_dim:
            sigmas = sigmas.unsqueeze(-1

        return sigmas        

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        # forward ODE
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
        # reverse ODE
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def _get_step_indices(self, timesteps: torch.Tensor) -> torch.Tensor:
        # ensure same dtype/device
        timesteps = timesteps.to(device=self._timesteps_sorted.device, dtype=self._timesteps_sorted.dtype)

        if self._timesteps_flipped:
            t = torch.flip(timesteps, dims=[0])
        else:
            t = timesteps

        idx = torch.searchsorted(self._timesteps_sorted, t)

        if self._timesteps_flipped:
            idx = (len(self.timesteps) - 1) - idx

        return idx

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type='linear',
        latents=None,
        patch_size=1
    ):
        self.timestep_type = timestep_type
        if timestep_type == 'linear' or timestep_type == 'weighted':
            timesteps = torch.linspace(1000, 1, num_timesteps, device=device)
            self.timesteps = timesteps
            # MPS optimization: pre-compute sorted base and flipped flag
            self._timesteps_sorted = torch.flip(self.timesteps, dims=[0])
            self._timesteps_flipped = True
            
            return timesteps
        elif timestep_type == 'sigmoid':
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = ((1 - t) * 1000)

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)
            # MPS optimization: pre-compute sorted base and flipped flag
            if self.timesteps[0] > self.timesteps[-1]:
                self._timesteps_sorted = torch.flip(self.timesteps, dims=[0])
                self._timesteps_flipped = True
            else:
                self._timesteps_sorted = self.timesteps
                self._timesteps_flipped = False

            return timesteps
        elif timestep_type in ['flux_shift', 'lumina2_shift', 'shift']:
            # matches inference dynamic shifting
            timesteps = torch.linspace(
                self._sigma_to_t(self.sigma_max),
                self._sigma_to_t(self.sigma_min),
                num_timesteps,
                device=device,
                dtype=torch.float32
            )            

            sigmas = timesteps / self.config.num_train_timesteps

            if self.config.use_dynamic_shifting:
                if latents is None:
                    raise ValueError('latents is None')

                # for flux we double up the patch size before sending her to simulate the latent reduction
                h = latents.shape[2]
                w = latents.shape[3]
                image_seq_len = h * w // (patch_size**2)

                mu = calculate_shift(
                    image_seq_len,
                    self.config.get("base_image_seq_len", 256),
                    self.config.get("max_image_seq_len", 4096),
                    self.config.get("base_shift", 0.5),
                    self.config.get("max_shift", 1.16),
                )
                sigmas = self.time_shift(mu, 1.0, sigmas)
            else:
                sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

            if self.config.shift_terminal:
                sigmas = self.stretch_shift_to_terminal(sigmas)

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)

            sigmas = torch.from_numpy(sigmas).to(
                dtype=torch.float32, device=device)

            if self.config.invert_sigmas:
                sigmas = 1.0 - sigmas

            timesteps = sigmas * self.config.num_train_timesteps

            # keep old "extra sigma" behavior, but align lengths
            sigmas = torch.cat([sigmas, sigmas[-1:]])
            timesteps = torch.cat([timesteps, timesteps[-1:]])

            # ensure monotonic descending timesteps and aligned sigmas
            order = torch.argsort(timesteps, descending=True)
            timesteps = timesteps[order]
            sigmas = sigmas[order]

            self.timesteps = timesteps
            self.sigmas = sigmas
            # MPS optimization: pre-compute sorted base and flipped flag
            if self.timesteps[0] > self.timesteps[-1]:
                self._timesteps_sorted = torch.flip(self.timesteps, dims=[0])
                self._timesteps_flipped = True
            else:
                self._timesteps_sorted = self.timesteps
                self._timesteps_flipped = False
            return timesteps

        elif timestep_type == 'lognorm_blend':
            # disgtribute timestepd to the center/early and blend in linear
            alpha = 0.75

            lognormal = LogNormal(loc=0, scale=0.333)

            # Sample from the distribution
            t1 = lognormal.sample((int(num_timesteps * alpha),)).to(device)

            # Scale and reverse the values to go from 1000 to 0
            t1 = ((1 - t1/t1.max()) * 1000)

            # add half of linear
            t2 = torch.linspace(1000, 1, int(
                num_timesteps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device=device)
            # MPS optimization: pre-compute sorted base and flipped flag
            if self.timesteps[0] > self.timesteps[-1]:
                self._timesteps_sorted = torch.flip(self.timesteps, dims=[0])
                self._timesteps_flipped = True
            else:
                self._timesteps_sorted = self.timesteps
                self._timesteps_flipped = False
            return timesteps
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")
