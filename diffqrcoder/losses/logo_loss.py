import torch
import torch.nn.functional as F
from torch import nn
from .perceptual_loss import VGGFeatureExtractor

class LogoLoss(nn.Module):
    def __init__(self,  
        feature_layer: int = 34,        # 这个参数现在不会传递给VGGFeatureExtractor  
        use_bn: bool = False,           # 这个参数现在不会传递给VGGFeatureExtractor  
        use_input_norm: bool = True,    # 这个参数现在不会传递给VGGFeatureExtractor  
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    ):  
        super().__init__()  
          
        # 使用现有VGGFeatureExtractor的正确参数  
        self.feature_extractor = VGGFeatureExtractor(  
            requires_grad=False,  
            pretrained_weights="DEFAULT"  
        ).to(device)

    def forward(self,
        generated_image: torch.Tensor,
        logo_image: torch.Tensor,
        logo_mask: torch.Tensor
    ) -> torch.Tensor:
        # 提取logo区域特征
        logo_region = generated_image * logo_mask
        target_region = logo_image * logo_mask

        # 计算特征相似性损失
        generated_features = self.feature_extractor(logo_region)
        target_features = self.feature_extractor(target_region)

        loss = 0
        for gen_feat, tar_feat in zip(generated_features, target_features):
            loss += F.mse_loss(gen_feat, tar_feat)

        return loss / len(generated_features)
