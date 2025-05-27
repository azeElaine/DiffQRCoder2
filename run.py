import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from diffusers import ControlNetModel, DDIMScheduler
from diffusers.utils import load_image

from diffqrcoder import DiffQRCoderPipeline


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--controlnet_ckpt",
        type=str,
        default="control_v1p_sd15_qrcode_monster"
    )
    parser.add_argument(
        "--pipe_ckpt",
        type=str,
        default="cetus-mix/cetusMix_Whalefall2_fp16.safetensors"
    )
    parser.add_argument(
        "--qrcode_path",
        type=str,
        default="qrcode/lywx.png"
    )
    parser.add_argument(
        "--qrcode_module_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--qrcode_padding",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Winter wonderland, fresh snowfall, evergreen trees, cozy log cabin, smoke rising from chimney, aurora borealis in night sky.",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="easynegative"
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "-srg",
        "--scanning_robust_guidance_scale",
        type=float,
        default=300,
    )
    parser.add_argument(
        "-pg",
        "--perceptual_guidance_scale",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--srmpgd_num_iteration",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--srmpgd_lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output"
    )
    # 在这里添加新的 logo 参数  
    parser.add_argument(  
        "--logo_path",  
        type=str,  
        default="qrcode/OIP-C.jpg",  
        help="Path to logo image file"  
    )  
    parser.add_argument(  
        "--logo_guidance_scale",   
        type=int,  
        default=50,  
        help="Scale for logo guidance loss"  
    )  
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)

    qrcode = load_image(args.qrcode_path)
    logo_img = load_image(args.logo_path)
    if args.logo_path:  
        logo_img = load_image(args.logo_path)  
        # 调整logo图像大小以减少内存使用  
        logo_img = logo_img.resize((256, 256))  # 限制logo尺寸 
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_ckpt,
        torch_dtype=torch.float16,
        local_files_only=True 
    )
    pipe = DiffQRCoderPipeline.from_single_file(
        args.pipe_ckpt,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        local_files_only=True 
    )
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # Memory optimizations  
    try:  
        pipe.enable_attention_slicing()  
        pipe.enable_xformers_memory_efficient_attention()  
    except AttributeError:  
    # Fallback if xformers is not available  
        pipe.enable_attention_slicing()  
      
    try:  
        pipe.enable_sequential_cpu_offload()  
    except AttributeError:  
        pass  
  
# Enable gradient checkpointing if available  
    if hasattr(pipe.unet, 'enable_gradient_checkpointing'):  
        pipe.unet.enable_gradient_checkpointing()

    result = pipe(
        prompt=args.prompt,
        qrcode=qrcode,
        logo_image=logo_img,
        logo_guidance_scale=args.logo_guidance_scale,  # 新增  
        qrcode_module_size=args.qrcode_module_size,
        qrcode_padding=args.qrcode_padding,
        negative_prompt=args.neg_prompt,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=args.device).manual_seed(1),
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        scanning_robust_guidance_scale=args.scanning_robust_guidance_scale,
        perceptual_guidance_scale=args.perceptual_guidance_scale,
        srmpgd_num_iteration=args.srmpgd_num_iteration,
        srmpgd_lr=args.srmpgd_lr,
    )
    result.images[0].save(Path(args.output_folder, "aesthetic_qrcode2.png"))
