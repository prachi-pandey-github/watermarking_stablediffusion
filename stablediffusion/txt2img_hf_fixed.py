#!/usr/bin/env python3
"""
Text-to-Image Generation with HuggingFace Diffusers and Watermarking
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
import os
import sys
import numpy as np
from PIL import Image

# Add current directory to path for local imports
sys.path.append('.')

def load_model(model_id, auth_token=None):
    """Load Stable Diffusion model from HuggingFace"""
    print(f"Loading model from Hugging Face: {model_id}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline
    if auth_token:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_auth_token=auth_token
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    
    pipe = pipe.to(device)
    
    if device == "cuda":
        print("Model loaded on GPU")
    else:
        print("Model loaded on CPU")
    
    return pipe

def gs_watermark_init_noise(args, watermark_message):
    """Initialize watermarked noise (simplified version for HuggingFace integration)"""
    try:
        from gs_insert import gs_watermark_init_noise as gs_watermark_func
        return gs_watermark_func(args, watermark_message)
    except ImportError:
        print("Warning: gs_insert.py not found, continuing without watermarking")
        return None

def apply_watermark_to_noise(latents, watermark_noise, strength=0.1):
    """Apply watermark noise to the initial latents"""
    if watermark_noise is not None and latents.shape == watermark_noise.shape:
        # Apply watermark with specified strength
        watermarked_latents = latents + strength * watermark_noise
        return watermarked_latents
    return latents

def create_watermarked_generator(pipe, args, watermark_message, key_hex, nonce_hex):
    """Create a generation function with watermarking support"""
    
    # Prepare watermarking if parameters are provided
    watermark_enabled = False
    watermark_noise = None
    
    if watermark_message and key_hex and nonce_hex:
        print(f"Watermarking enabled: '{watermark_message}'")
        
        # Create a minimal args object for gs_watermark_init_noise
        class WatermarkArgs:
            def __init__(self):
                self.key_hex = key_hex
                self.nonce_hex = nonce_hex
                self.H = args.height
                self.W = args.width
        
        watermark_args = WatermarkArgs()
        
        try:
            watermark_noise = gs_watermark_init_noise(watermark_args, watermark_message)
            if watermark_noise is not None:
                # Convert to torch tensor if it's numpy
                if isinstance(watermark_noise, np.ndarray):
                    watermark_noise = torch.from_numpy(watermark_noise).float()
                
                # Move to same device as pipe
                watermark_noise = watermark_noise.to(pipe.device)
                watermark_enabled = True
                print("Watermarked noise generated successfully")
            else:
                print("Warning: Watermark generation failed")
        except Exception as e:
            print(f"Warning: Watermarking failed: {e}")
            print("Continuing without watermarking...")
    else:
        print("Warning: missing parameters (key/nonce) or gs_insert.py")
    
    def generate_image(prompt, **kwargs):
        print(f"Generating image: '{prompt[:50]}...'")
        
        # Generate with or without watermarking
        if watermark_enabled and watermark_noise is not None:
            # Custom generation with watermarked initial noise
            generator = torch.Generator(device=pipe.device)
            if args.seed is not None:
                generator.manual_seed(args.seed)
            
            # Get the scheduler's init_noise_sigma
            init_noise_sigma = pipe.scheduler.init_noise_sigma
            
            # Generate initial latents
            shape = (1, pipe.unet.config.in_channels, args.height // 8, args.width // 8)
            latents = torch.randn(shape, generator=generator, device=pipe.device, dtype=pipe.unet.dtype)
            latents = latents * init_noise_sigma
            
            # Apply watermark to latents
            if watermark_noise.shape == latents.shape:
                latents = apply_watermark_to_noise(latents, watermark_noise, strength=0.1)
            
            # Generate image with watermarked latents
            image = pipe(prompt, latents=latents, **kwargs).images[0]
        else:
            # Standard generation
            generator = torch.Generator(device=pipe.device)
            if args.seed is not None:
                generator.manual_seed(args.seed)
            image = pipe(prompt, generator=generator, **kwargs).images[0]
        
        print("Image generated successfully")
        return image, watermark_enabled
    
    return generate_image

def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion and optional watermarking")
    
    # Image generation arguments
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output.png", help="Output image filename")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1", 
                       help="HuggingFace model ID")
    
    # Image parameters
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height") 
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # Watermarking arguments
    parser.add_argument("--message", type=str, default="", help="Watermark message")
    parser.add_argument("--key_hex", type=str, default="", help="Watermark key (hex)")
    parser.add_argument("--nonce_hex", type=str, default="", help="Watermark nonce (hex)")
    
    # HuggingFace arguments
    parser.add_argument("--auth_token", type=str, default=None, help="HuggingFace auth token")
    
    args = parser.parse_args()
    
    print("Starting Stable Diffusion with Hugging Face Integration")
    print(f"Prompt: {args.prompt}")
    print(f"Model: {args.model}")
    print(f"Size: {args.width}x{args.height}")
    
    try:
        # Load model from Hugging Face
        pipe = load_model(args.model, args.auth_token)
        
        # Create watermarked generator
        generate_image = create_watermarked_generator(
            pipe, args, args.message, args.key_hex, args.nonce_hex
        )
        
        # Generate image
        generation_kwargs = {
            "num_inference_steps": args.steps,
            "guidance_scale": args.cfg_scale,
            "width": args.width,
            "height": args.height
        }
        
        image, watermark_enabled = generate_image(args.prompt, **generation_kwargs)
        
        # Save image
        image.save(args.output)
        print(f"Image saved to: {args.output}")
        
        # Status message
        if watermark_enabled:
            print("Watermark embedded successfully!")
            print(f"Message: {args.message}")
            print("Use extract.py to verify the watermark")
        else:
            print("Image generated without watermarking")
        
        print("Generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
