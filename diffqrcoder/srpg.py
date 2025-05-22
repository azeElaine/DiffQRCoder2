import torch
from torch import nn

from diffqrcoder.losses import PerceptualLoss, ScanningRobustLoss



GRADIENT_SCALE = 100


class ScanningRobustPerceptualGuidance(nn.Module):
    def __init__(
        self,
        module_size: int = 20,
        scanning_robust_guidance_scale: int = 500,
        perceptual_guidance_scale: int = 2,
        logo_guidance_scale: int = 100,
        feature_layers: list = [3, 8, 15, 22],
        use_normalize: bool = True
    ):
        super().__init__()
        self.module_size = module_size
        self.scanning_robust_guidance_scale = scanning_robust_guidance_scale
        self.perceptual_guidance_scale = perceptual_guidance_scale
        self.logo_guidance_scale = logo_guidance_scale
        self.scanning_robust_loss_fn = ScanningRobustLoss(module_size=module_size)
        self.perceptual_loss_fn = PerceptualLoss()
        self.logo_loss_fn = LogoLoss(feature_layers=feature_layers, use_normalize=use_normalize)

    def compute_loss(self,
        image: torch.Tensor,
        qrcode: torch.Tensor,
        ref_image: torch.Tensor,
        logo_image: torch.Tensor = None,
        logo_mask: torch.Tensor = None) -> torch.Tensor:
        loss = (
            self.scanning_robust_guidance_scale * self.scanning_robust_loss_fn(image, qrcode) +
            self.perceptual_guidance_scale * self.perceptual_loss_fn(image, ref_image)
        )
        
        # Add logo loss if provided
        if logo_image is not None and logo_mask is not None:
            # Ensure mask is in correct format
            if len(logo_mask.shape) == 3:
                logo_mask = logo_mask.unsqueeze(1)
            logo_mask = logo_mask.to(image.dtype)
            
            logo_loss = self.logo_loss_fn(image, logo_image, logo_mask)
            loss += self.logo_guidance_scale * logo_loss
            
        return loss * GRADIENT_SCALE

    def compute_score(self, latents: torch.Tensor, image: torch.Tensor, qrcode: torch.Tensor,
                     ref_image: torch.Tensor, logo_image: torch.Tensor = None,
                     logo_mask: torch.Tensor = None) -> torch.Tensor:
        loss = self.compute_loss(image, qrcode, ref_image, logo_image, logo_mask)
        return -torch.autograd.grad(loss, latents)[0]
