#!/usr/bin/env python3
"""
02_generate_images.py

Generate images using three open-source text-to-image models:
1. Stable Diffusion XL (SDXL)
2. DeepFloyd IF
3. Playground-v2.5

Generates 1 image per prompt per model = 200 images per model = 600 total
"""

import os
import argparse
import time
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from diffusers import DiffusionPipeline
from typing import Dict, List
import json
import gc


def sanitize_filename(text: str) -> str:
    """Create safe filename from text."""
    import re
    s = re.sub(r'[^\w\s]', '_', text)
    return s.replace(' ', '_')[:100]


def load_prompts(csv_path: str) -> Dict[str, List[Dict]]:
    """
    Load prompts from CSV and organize by character.
    Returns: {character: [{'prompt': str, 'label': str}, ...]}
    """
    df = pd.read_csv(csv_path)
    characters = {}
    
    for _, row in df.iterrows():
        target = row['target']
        if target not in characters:
            characters[target] = []
        
        characters[target].append({
            'prompt': row['prompt'],
            'label': row['prompt_label'],
            'type': row['prompt_type']
        })
    
    return characters


class ImageGenerator:
    """Base class for image generation with different models."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.pipe = None
    
    def generate(self, prompt: str) -> 'Image':
        raise NotImplementedError
    
    def cleanup(self):
        """Clean up model from memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()


class SDXLGenerator(ImageGenerator):
    """Stable Diffusion XL generator."""
    
    def __init__(self, device='cuda'):
        super().__init__(device)
        print("Loading Stable Diffusion XL...")
        self.pipe = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16'
        ).to(device)
        
        # Enable memory efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xFormers not available, using default attention")
    
    def generate(self, prompt: str):
        return self.pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=3,
            height=768,
            width=768
        ).images[0]


class DeepFloydGenerator(ImageGenerator):
    """DeepFloyd IF generator (two-stage)."""
    
    def __init__(self, device='cuda'):
        super().__init__(device)
        print("Loading DeepFloyd IF Stage I...")
        self.pipe1 = DiffusionPipeline.from_pretrained(
            'DeepFloyd/IF-I-XL-v1.0',
            variant='fp16',
            torch_dtype=torch.float16
        ).to(device)
        self.pipe1.enable_model_cpu_offload()
        
        print("Loading DeepFloyd IF Stage II...")
        self.pipe2 = DiffusionPipeline.from_pretrained(
            'DeepFloyd/IF-II-L-v1.0',
            text_encoder=None,
            variant='fp16',
            torch_dtype=torch.float16
        ).to(device)
        self.pipe2.enable_model_cpu_offload()
    
    def generate(self, prompt: str):
        # Stage I: Generate 64x64
        prompt_embeds, negative_embeds = self.pipe1.encode_prompt(prompt)
        stage1_output = self.pipe1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type='pt'
        ).images
        
        # Stage II: Upscale to 256x256
        image = self.pipe2(
            image=stage1_output,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type='pil'
        ).images[0]
        
        return image
    
    def cleanup(self):
        """Clean up both pipeline stages."""
        if hasattr(self, 'pipe1'):
            del self.pipe1
        if hasattr(self, 'pipe2'):
            del self.pipe2
        gc.collect()
        torch.cuda.empty_cache()


class PlaygroundGenerator(ImageGenerator):
    """Playground-v2.5 generator."""
    
    def __init__(self, device='cuda'):
        super().__init__(device)
        print("Loading Playground-v2.5...")
        self.pipe = DiffusionPipeline.from_pretrained(
            'playgroundai/playground-v2.5-1024px-aesthetic',
            torch_dtype=torch.float16
        ).to(device)
    
    def generate(self, prompt: str):
        return self.pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=3
        ).images[0]


def get_generator(model_name: str, device: str) -> ImageGenerator:
    """Factory function to get the appropriate generator."""
    generators = {
        'sdxl': SDXLGenerator,
        'deepfloyd': DeepFloydGenerator,
        'playground': PlaygroundGenerator
    }
    
    if model_name not in generators:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(generators.keys())}")
    
    return generators[model_name](device)


def generate_images_for_model(
    model_name: str,
    generator: ImageGenerator,
    prompts_dict: Dict[str, List[Dict]],
    output_dir: Path,
    sleep_time: float = 0.5
):
    """Generate images for a single model."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Track generation for resuming
    generated_log = model_dir / 'generated.txt'
    generated_set = set()
    if generated_log.exists():
        with open(generated_log, 'r') as f:
            generated_set = set(line.strip() for line in f)
    
    # Create metadata file
    metadata_file = model_dir / 'metadata.json'
    metadata = []
    
    total_prompts = sum(len(prompts) for prompts in prompts_dict.values())
    
    with tqdm(total=total_prompts, desc=f"Generating {model_name}") as pbar:
        for character, prompts_data in prompts_dict.items():
            char_dir = model_dir / sanitize_filename(character)
            char_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, entry in enumerate(prompts_data, start=1):
                prompt = entry['prompt']
                label = entry['label']
                prompt_type = entry['type']
                
                # Create unique identifier
                image_id = f"{character}_{idx}_{label}"
                
                # Skip if already generated
                if image_id in generated_set:
                    pbar.update(1)
                    continue
                
                # Create filename
                filename = f"{sanitize_filename(character)}_{idx:03d}_{sanitize_filename(label)}.png"
                image_path = char_dir / filename
                
                try:
                    # Generate image
                    image = generator.generate(prompt)
                    image.save(image_path)
                    
                    # Save metadata
                    metadata.append({
                        'image_id': image_id,
                        'filename': str(image_path.relative_to(model_dir)),
                        'character': character,
                        'prompt': prompt,
                        'prompt_label': label,
                        'prompt_type': prompt_type,
                        'prompt_index': idx
                    })
                    
                    # Log successful generation
                    with open(generated_log, 'a') as f:
                        f.write(f"{image_id}\n")
                    
                except Exception as e:
                    print(f"\nError generating {character}/{label}: {e}")
                
                # Sleep to avoid rate limits
                time.sleep(sleep_time)
                pbar.update(1)
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate images for PC-D experiments')
    parser.add_argument('--prompts_csv', type=str, required=True,
                        help='Path to CSV with all prompts')
    parser.add_argument('--output_dir', type=str, default='data/images',
                        help='Output directory for images')
    parser.add_argument('--models', nargs='+', 
                        default=['playground', 'sdxl', 'deepfloyd'],
                        help='Models to use for generation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--sleep_time', type=float, default=0.5,
                        help='Sleep time between generations')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    print("Loading prompts...")
    prompts_dict = load_prompts(args.prompts_csv)
    
    # Summary
    total_prompts = sum(len(prompts) for prompts in prompts_dict.values())
    print(f"\nLoaded {total_prompts} prompts for {len(prompts_dict)} characters")
    print(f"Will generate {total_prompts * len(args.models)} total images")
    
    # Generate images for each model
    for model_name in args.models:
        print(f"\n{'=' * 50}")
        print(f"Starting generation with {model_name}")
        print(f"{'=' * 50}")
        
        # Initialize generator
        generator = get_generator(model_name, args.device)
        
        # Generate images
        generate_images_for_model(
            model_name=model_name,
            generator=generator,
            prompts_dict=prompts_dict,
            output_dir=output_dir,
            sleep_time=args.sleep_time
        )
        
        # Clean up to free memory
        generator.cleanup()
        print(f"Completed {model_name}")
    
    print("\n=== Generation Complete ===")
    print(f"Total images generated: {total_prompts * len(args.models)}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
