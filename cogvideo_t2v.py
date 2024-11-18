
import torch
from diffusers import CogVideoXPipeline
import matplotlib.pyplot as plt
import argparse
import imageio
from utils import (
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
    save_by_timesteps
)

def arg_parse():
    parser = argparse.ArgumentParser(description="Generate videos using CogVideoXPipeline with configurable parameters.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the pipeline (e.g., 'cuda:0').")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video.")
    parser.add_argument("--width", type=int, default=720, help="Width of the output video.")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames in the output video.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=6, help="Guidance scale for video generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate the video.")
    parser.add_argument("--output_path", type=str, default="test.mp4", help="Path to save the generated video.")
    parser.add_argument("--attn_map_output_dir", type=str, default="vis", help="Directory to save attention maps.")
    parser.add_argument("--save_tensor_dir", type=str, default=None, help="Directory to save attention maps.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for the output video.")
    args = parser.parse_args()    
    return args

def main(args):
    
    cross_attn_init()
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(args.device)
    pipe.transformer = set_layer_with_name_and_path(pipe.transformer)
    pipe.transformer = register_cross_attention_hook(pipe.transformer)

    attn_height = args.height // 16
    attn_width = args.width // 16
    latent_frame = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1

    video = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_videos_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
    ).frames[0]

    imageio.mimsave(args.output_path, video, fps=args.fps)
    print(f"Generated video saved to {args.output_path}")

    save_by_timesteps(latent_frame, attn_height, attn_width, args.attn_map_output_dir,args.save_tensor_dir)
    print(f"Attention maps saved to {args.attn_map_output_dir}")

if __name__ == "__main__":
    args = arg_parse()
    main(args)