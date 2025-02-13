import torch
import imageio
import os
import argparse
from PIL import Image
from einops import rearrange
import numpy as np
from torchvision.transforms import Lambda
from torchvision import transforms

from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer

from allegro.pipelines.pipeline_allegro_ti2v import AllegroTI2VPipeline
from allegro.pipelines.data_process import ToTensorVideo, CenterCropResizeVideo
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro_ti2v import AllegroTransformerTI2V3DModel

from torchvision.utils import save_image

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Inverse of Lambda(lambda x: 2. * x - 1.),
    which maps [0,1] -> [-1,1]. So do: (x + 1)/2
    to map back from [-1,1] -> [0,1].
    """
    return (tensor + 1.0) / 2.0


def preprocess_images(first_frame, last_frame, height, width, device, dtype, output_path):
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    transform = transforms.Compose([
        ToTensorVideo(),
        CenterCropResizeVideo((height, width), True),
        norm_fun
    ])
    images = []
    if first_frame is not None and len(first_frame.strip()) != 0: 
        print("first_frame:", first_frame)
        images.append(first_frame)
    else:
        print("ERROR: First frame must be provided in Allegro-TI2V!")
        raise NotImplementedError
    if last_frame is not None and len(last_frame.strip()) != 0: 
        print("last_frame:", last_frame)
        images.append(last_frame)

    if len(images) == 1:    # first frame as condition
        print("Video generation with given first frame.")
        conditional_images_indices = [0]
    elif len(images) == 2:  # first&last frames as condition
        print("Video generation with given first and last frame.")
        conditional_images_indices = [0, -1]
    else:
        print("ERROR: Only support 1 or 2 conditional images!")
        raise NotImplementedError
    
    try:
        conditional_images = [Image.open(image).convert("RGB") for image in images]
        # write conditional image to disk
        for i, img in enumerate(conditional_images):
            img.save(f'{os.path.dirname(output_path)}/conditional_image.png')
        
        conditional_images = [torch.from_numpy(np.copy(np.array(image))) for image in conditional_images]
        conditional_images = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in conditional_images]
        conditional_images = [transform(image).to(device=device, dtype=dtype) for image in conditional_images]
        transformed_tensors = [t.to(device=device, dtype=dtype) for t in conditional_images]
        for i, t in enumerate(transformed_tensors):
            # t shape is (1, C, H, W)
            # Denormalize
            denorm_t = denormalize(t).clamp(0, 1)
            
            # Option A: Use torchvision's save_image:
            transformed_path = os.path.join(
                os.path.dirname(output_path),
                f'conditional_image_{i}_transformed.png'
            )
            save_image(denorm_t, transformed_path)
            print(f"Saved transformed image to {transformed_path}")
            
    except Exception as e:
        print('Error when loading images')
        print(f'condition images are {images}')
        raise e

    return dict(conditional_images=conditional_images, conditional_images_indices=conditional_images_indices)

def prompt_formatting(user_prompt, positive_prompt,):
    if user_prompt is None:
        print("ERROR: User prompt must be provided in Allegro-TI2V!")
        raise NotImplementedError
    user_prompt = user_prompt.lower().strip()
    if user_prompt == '' or len(user_prompt) == 0:
        print("ERROR: User prompt must be provided in Allegro-TI2V!")
        raise NotImplementedError
    user_prompt = positive_prompt.format(user_prompt)
    
    return user_prompt


def single_inference(args):
    dtype=torch.bfloat16

    # vae have better formance in float32
    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32).cuda()
    vae.eval()

    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder, 
        torch_dtype=dtype
    )
    text_encoder.eval()

    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer,
    )

    scheduler = EulerAncestralDiscreteScheduler()

    transformer = AllegroTransformerTI2V3DModel.from_pretrained(
        args.dit,
        torch_dtype=dtype
    ).cuda()
    transformer.eval()   

    allegro_ti2v_pipeline = AllegroTI2VPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer
    ).to("cuda:0")


    positive_prompt = """
(masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
{} 
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
sharp focus, high budget, cinemascope, moody, epic, gorgeous
"""

    negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""
    user_prompt = prompt_formatting(args.user_prompt, positive_prompt)
    pre_results = preprocess_images(args.first_frame, args.last_frame, height=720, width=900, device=torch.cuda.current_device(), dtype=torch.bfloat16, output_path=args.output_path)
    cond_imgs = pre_results['conditional_images']
    cond_imgs_indices = pre_results['conditional_images_indices']

    if args.enable_cpu_offload:
        allegro_ti2v_pipeline.enable_sequential_cpu_offload()
        print("cpu offload enabled")
        
    out_video = allegro_ti2v_pipeline(
        user_prompt, 
        negative_prompt=negative_prompt,
        conditional_images=cond_imgs,
        conditional_images_indices=cond_imgs_indices,
        num_frames=88,
        height=720,
        width=1280,
        num_inference_steps=args.num_sampling_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=512,
        generator=torch.Generator(device="cuda:0").manual_seed(args.seed),
    ).video[0]

    imageio.mimwrite(args.output_path, out_video, fps=15, quality=6)  # highest quality is 10, lowest is 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_prompt", type=str, default='')
    parser.add_argument('--first_frame', type=str, default='', help='A single image file as the first frame.')
    parser.add_argument('--last_frame', type=str, default='', help='A single image file as the last frame.')
    parser.add_argument("--vae", type=str, default='')
    parser.add_argument("--dit", type=str, default='')
    parser.add_argument("--text_encoder", type=str, default='')
    parser.add_argument("--tokenizer", type=str, default='')
    parser.add_argument("--output_path", type=str, default="./output_videos/test_video.mp4")
    parser.add_argument("--guidance_scale", type=float, default=8)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1427329220)
    parser.add_argument("--enable_cpu_offload", action='store_true')
    parser.add_argument("--prompt_path", type=str, default='', required=True)

    args = parser.parse_args()

    with open(args.prompt_path, "r") as file:
        content = file.read().strip()
        prompt, img_path = content.split("@@")
        args.user_prompt = prompt.strip()
        args.first_frame = os.path.expandvars(img_path.strip())
    
    args.output_path = os.path.join(args.output_path, "output.mp4")
    output_dir = os.path.dirname(args.output_path)

    from pathlib import Path

    # Ensure output directory exists
    output_dir = Path(output_dir)  # Convert string to Path object
    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import json

    config_path = output_dir / "config.json"  # This now works with Path
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
        
    # Proceed with single inference
    single_inference(args)
