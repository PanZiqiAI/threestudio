from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import IFPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("deep-floyd-guidance")
class DeepFloydGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "DeepFloyd/IF-I-XL-v1.0"
        # FIXME: xformers error
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = True
        guidance_scale: float = 20.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        weighting_strategy: str = "sds"

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Deep Floyd ...")

        self.weights_dtype = torch.float16 if self.cfg.half_precision_weights else torch.float32
        ################################################################################################################
        # 构建模型.
        ################################################################################################################
        self.pipe = IFPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, text_encoder=None, safety_checker=None, watermarker=None,
            feature_extractor=None, requires_safety_checker=False, variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype).to(self.device)
        """ Deploy. """
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info("PyTorch2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                threestudio.warn("xformers is not available, memory efficient attention is not enabled.")
            else:
                threestudio.warn(f"Use DeepFloyd with xformers may raise error, see https://github.com/deep-floyd/IF/issues/52 to track this problem.")
                self.pipe.enable_xformers_memory_efficient_attention()
        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
        # --------------------------------------------------------------------------------------------------------------
        # UNet.
        # --------------------------------------------------------------------------------------------------------------
        self.unet = self.pipe.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        # --------------------------------------------------------------------------------------------------------------
        # Scheduler.
        # --------------------------------------------------------------------------------------------------------------
        self.scheduler = self.pipe.scheduler
        # --------------------------------------------------------------------------------------------------------------
        # Timesteps.
        # --------------------------------------------------------------------------------------------------------------
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        # --------------------------------------------------------------------------------------------------------------
        # Alphas.
        # --------------------------------------------------------------------------------------------------------------
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

        self.grad_clip_val: Optional[float] = None
        threestudio.info(f"Loaded Deep Floyd!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    ####################################################################################################################
    # Forward.
    ####################################################################################################################

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(self, latents: Float[Tensor, "..."], t: Float[Tensor, "..."], encoder_hidden_states: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """ UNet前向计算. """
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype), t.to(self.weights_dtype), encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype)
        ).sample.to(input_dtype)

    def __call__(
        self, rgb: Float[Tensor, "B H W C"], prompt_utils: PromptProcessorOutput, elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"], camera_distances: Float[Tensor, "B"], rgb_as_latents=False, guidance_eval=False, **kwargs):
        """
        :param rgb: (batch, h, w, c).
        :param prompt_utils:
        :param elevation: (batch, ).
        :param azimuth: (batch, ).
        :param camera_distances: (batch, ).
        :param rgb_as_latents: bool.
        :param guidance_eval: bool.
        """
        batch_size = rgb.shape[0]
        # --------------------------------------------------------------------------------------------------------------
        # RGB图像. (batch, c, h, w).
        # --------------------------------------------------------------------------------------------------------------
        # rgb_BCHW. (batch, c, h, w).
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        assert rgb_as_latents is False, f"No latent space in {self.__class__.__name__}"
        rgb_BCHW = rgb_BCHW * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        # latents. (batch, c, 64, 64).
        latents = F.interpolate(rgb_BCHW, (64, 64), mode="bilinear", align_corners=False)
        # --------------------------------------------------------------------------------------------------------------
        # 时间步. (batch, ).
        # --------------------------------------------------------------------------------------------------------------
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device)

        ################################################################################################################
        # 生成噪音(ground-truth)并预测(prediction).
        ################################################################################################################
        # --------------------------------------------------------------------------------------------------------------
        # 使用负面提示词
        # --------------------------------------------------------------------------------------------------------------
        if prompt_utils.use_perp_neg:
            text_embeddings, neg_guidance_weights = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting)
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(latent_model_input, torch.cat([t] * 4), encoder_hidden_states=text_embeddings)  # (4B, 6, 64, 64)

            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(3, dim=1)
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(-1, 1, 1, 1) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (e_pos + accum_grad)
        # --------------------------------------------------------------------------------------------------------------
        # 只使用正面提示词
        # --------------------------------------------------------------------------------------------------------------
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting)
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(latent_model_input, torch.cat([t] * 2), encoder_hidden_states=text_embeddings)  # (2B, 6, 64, 64)

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {"loss_sds": loss_sds, "grad_norm": grad.norm(), "min_step": self.min_step, "max_step": self.max_step}

        if guidance_eval:
            guidance_eval_utils = {
                "use_perp_neg": prompt_utils.use_perp_neg, "neg_guidance_weights": neg_guidance_weights,
                "text_embeddings": text_embeddings, "t_orig": t, "latents_noisy": latents_noisy,
                "noise_pred": torch.cat([noise_pred, predicted_variance], dim=1)}
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances):
                texts.append(f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}")
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(self, latents_noisy, t, text_embeddings, use_perp_neg=False, neg_guidance_weights=None):
        """
        :param latents_noisy: (1, 3, 64, 64).
        :param t: (, ).
        :param text_embeddings: (2或4, 77, 4096).
        :param use_perp_neg:
        :param neg_guidance_weights: (1, 2).
        """
        batch_size = latents_noisy.shape[0]
        if use_perp_neg:
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input, torch.cat([t.reshape(1)] * 4).to(self.device), encoder_hidden_states=text_embeddings)  # (4B, 6, 64, 64)
            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(3, dim=1)
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(-1, 1, 1, 1) * perpendicular_component(e_i_neg, e_pos)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (e_pos + accum_grad)
        else:
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input, torch.cat([t.reshape(1)] * 2).to(self.device), encoder_hidden_states=text_embeddings)  # (2B, 6, 64, 64)
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        return torch.cat([noise_pred, predicted_variance], dim=1)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(self, t_orig, text_embeddings, latents_noisy, noise_pred, use_perp_neg=False, neg_guidance_weights=None):
        """
        :param t_orig: (batch, ). torch@long. 时间步.
        :param text_embeddings: (2或4*batch, 77, 4096)，其中N对应pos/uncond(/neg1/neg2)
        :param latents_noisy: (batch, 3, 64, 64). 加噪后的latents.
        :param noise_pred: (batch, 6, 64, 64). 最终预测噪音（融合正负后） + 预测方差.
        :param use_perp_neg:
        :param neg_guidance_weights: None or (batch, N-2). 对应于neg1/neg2
        """
        bs = min(self.cfg.max_items_eval, latents_noisy.shape[0]) if self.cfg.max_items_eval > 0 else latents_noisy.shape[0]

        # --------------------------------------------------------------------------------------------------------------
        # 设置时间步：use only 50 timesteps
        # --------------------------------------------------------------------------------------------------------------
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device) # 均等分1000到0,有50个数据 (间隔为20).
        # --------------------------------------------------------------------------------------------------------------
        # 找到小于且最接近于t_orig的timestep. (batch, ).
        # --------------------------------------------------------------------------------------------------------------
        # (batch, 50). 表明每一个timestep是否大于t_orig.
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[:bs].unsqueeze(-1)  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1).indices
        # (batch, ). 找到小于且最接近于t_orig的timestep.
        t = self.scheduler.timesteps_gpu[idxs]
        """ 计算fraction. """
        frac = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())

        # --------------------------------------------------------------------------------------------------------------
        # 获取imgs_1step & imgs_1orig. (batch, 3, 64, 64).
        """
        @imgs_1step: 最接近的上一个timestep的去噪结果.
        @imgs_1orig: 根据预测噪音推断的原始图像（对应于t0时间步）. """
        # --------------------------------------------------------------------------------------------------------------
        latents_1step, pred_1orig = [], []
        for b in range(bs):
            step_output = self.scheduler.step(noise_pred[b:b + 1], t[b], latents_noisy[b:b + 1])
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = (latents_1step / 2 + 0.5).permute(0, 2, 3, 1)
        imgs_1orig = (pred_1orig / 2 + 0.5).permute(0, 2, 3, 1)

        # --------------------------------------------------------------------------------------------------------------
        # 获取逐步去噪的最终去噪结果. (batch, 3, 64, 64). 
        # --------------------------------------------------------------------------------------------------------------
        latents_final = []
        for b, i in enumerate(idxs):
            # (1, 3, 64, 64). 当前样本的上一个timestep的去噪结果.
            latents = latents_1step[b:b+1]
            # (4或2, 77, 4096)       todo: 应该是2*len(idxs)+2b, 2*len(idxs)+2b+1
            text_emb = text_embeddings[[b, b+len(idxs), b+2*len(idxs), b+3*len(idxs)], ...] \
                if use_perp_neg else text_embeddings[[b, b+len(idxs)], ...]
            neg_guid = neg_guidance_weights[b:b+1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i+1:], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(latents, t, text_emb, use_perp_neg, neg_guid)
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
            latents_final.append(latents)
        latents_final = torch.cat(latents_final)
        imgs_final = (latents_final / 2 + 0.5).permute(0, 2, 3, 1)

        return {
            "noise_levels": frac,
            "imgs_noisy": (latents_noisy[:bs] / 2 + 0.5).permute(0, 2, 3, 1),
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
