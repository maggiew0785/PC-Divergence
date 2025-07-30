#!/usr/bin/env python3
"""
03_compute_embeddings.py

Compute OpenCLIP embeddings for prompts, character names, and generated images.
This script produces all embeddings needed for PC-D calculation:
1. Prompt embeddings
2. Character name embeddings  
3. Generated image embeddings

Uses OpenCLIP ViT-H/14 model for all embeddings.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import open_clip
from PIL import Image
import json
from typing import Dict, List, Tuple


def load_character_list(char_file: str) -> List[str]:
    """Load character names from text file."""
    with open(char_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_image_metadata(image_dir: Path, model_name: str) -> pd.DataFrame:
    """Load metadata for generated images from a specific model."""
    metadata_file = image_dir / model_name / 'metadata.json'
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata)
    df['model'] = model_name
    df['full_path'] = df['filename'].apply(lambda x: str(image_dir / model_name / x))
    
    return df


def compute_text_embeddings(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(embeddings)


def compute_image_embeddings(
    image_paths: List[str],
    model,
    preprocess,
    device: str,
    batch_size: int = 16
) -> np.ndarray:
    """Compute embeddings for a list of images."""
    embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                batch_images.append(preprocess(image))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Create zero embedding for failed images
                batch_images.append(torch.zeros_like(preprocess(Image.new('RGB', (224, 224)))))
        
        if batch_images:
            image_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
    
    return np.vstack(embeddings) if embeddings else np.array([])


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description='Compute OpenCLIP embeddings for PC-D')
    parser.add_argument('--image_dir', type=str, default='data/images',
                        help='Directory containing generated images')
    parser.add_argument('--prompts_csv', type=str, required=True,
                        help='Path to CSV with all prompts')
    parser.add_argument('--characters_file', type=str, default='data/characters.txt',
                        help='Text file with character names')
    parser.add_argument('--output_dir', type=str, default='results/embeddings',
                        help='Output directory for embeddings')
    parser.add_argument('--models', nargs='+', 
                        default=['playground', 'sdxl', 'deepfloyd'],
                        help='Models to process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load OpenCLIP model
    print("Loading OpenCLIP ViT-H/14...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', 
        pretrained='laion2b_s32b_b79k'
    )
    model = model.to(args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    
    # Load character names
    characters = load_character_list(args.characters_file)
    print(f"Loaded {len(characters)} characters")
    
    # Load prompts
    prompts_df = pd.read_csv(args.prompts_csv)
    print(f"Loaded {len(prompts_df)} prompts")
    
    # Compute character embeddings
    print("\nComputing character name embeddings...")
    char_embeddings = compute_text_embeddings(
        characters, model, tokenizer, args.device
    )
    
    # Save character embeddings
    char_emb_df = pd.DataFrame({
        'character': characters,
        'embedding': [emb.tolist() for emb in char_embeddings]
    })
    char_emb_df.to_csv(output_dir / 'character_embeddings.csv', index=False)
    
    # Compute prompt embeddings and similarities
    print("\nComputing prompt embeddings and character similarities...")
    unique_prompts = prompts_df['prompt'].unique().tolist()
    prompt_embeddings = compute_text_embeddings(
        unique_prompts, model, tokenizer, args.device, batch_size=args.batch_size
    )
    
    # Calculate prompt-character similarities for prompt specificity
    prompt_similarities = []
    for _, row in prompts_df.iterrows():
        prompt = row['prompt']
        character = row['target']
        
        # Find indices
        prompt_idx = unique_prompts.index(prompt)
        char_idx = characters.index(character) if character in characters else -1
        
        if char_idx >= 0:
            similarity = compute_cosine_similarity(
                prompt_embeddings[prompt_idx],
                char_embeddings[char_idx]
            )
        else:
            similarity = 0.0
        
        prompt_similarities.append({
            'target': character,
            'prompt': prompt,
            'prompt_label': row['prompt_label'],
            'prompt_type': row['prompt_type'],
            'prompt_char_similarity': similarity
        })
    
    # Save prompt similarities
    prompt_sim_df = pd.DataFrame(prompt_similarities)
    prompt_sim_df.to_csv(output_dir / 'prompt_character_similarities.csv', index=False)
    
    # Process each model's images
    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Processing {model_name} images...")
        print(f"{'='*50}")
        
        try:
            # Load image metadata
            metadata_df = load_image_metadata(Path(args.image_dir), model_name)
            print(f"Found {len(metadata_df)} images")
            
            # Compute image embeddings
            image_paths = metadata_df['full_path'].tolist()
            image_embeddings = compute_image_embeddings(
                image_paths, model, preprocess, args.device, batch_size=args.batch_size
            )
            
            # Calculate similarities for PC-D
            results = []
            for idx, row in metadata_df.iterrows():
                # Get embeddings
                img_emb = image_embeddings[idx]
                
                # Find prompt embedding
                prompt = row['prompt']
                prompt_idx = unique_prompts.index(prompt) if prompt in unique_prompts else -1
                
                # Find character embedding
                character = row['character']
                char_idx = characters.index(character) if character in characters else -1
                
                # Calculate similarities
                cos_img_prompt = compute_cosine_similarity(
                    img_emb, prompt_embeddings[prompt_idx]
                ) if prompt_idx >= 0 else 0.0
                
                cos_img_char = compute_cosine_similarity(
                    img_emb, char_embeddings[char_idx]
                ) if char_idx >= 0 else 0.0
                
                results.append({
                    'model': model_name,
                    'image_id': row['image_id'],
                    'filename': row['filename'],
                    'character': character,
                    'prompt': prompt,
                    'prompt_label': row['prompt_label'],
                    'prompt_type': row['prompt_type'],
                    'cos_img_prompt': cos_img_prompt,
                    'cos_img_char': cos_img_char,
                    'prompt_char_similarity': prompt_sim_df[
                        (prompt_sim_df['prompt'] == prompt) & 
                        (prompt_sim_df['target'] == character)
                    ]['prompt_char_similarity'].iloc[0] if len(prompt_sim_df[
                        (prompt_sim_df['prompt'] == prompt) & 
                        (prompt_sim_df['target'] == character)
                    ]) > 0 else 0.0
                })
            
            # Save results
            results_df = pd.DataFrame(results)
            output_file = output_dir / f'embeddings_{model_name}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"Saved embeddings to {output_file}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    # Create combined embeddings file
    print("\nCreating combined embeddings file...")
    all_embeddings = []
    for model_name in args.models:
        emb_file = output_dir / f'embeddings_{model_name}.csv'
        if emb_file.exists():
            df = pd.read_csv(emb_file)
            all_embeddings.append(df)
    
    if all_embeddings:
        combined_df = pd.concat(all_embeddings, ignore_index=True)
        combined_df.to_csv(output_dir / 'all_embeddings.csv', index=False)
        print(f"Saved combined embeddings to {output_dir / 'all_embeddings.csv'}")
    
    print("\n=== Embedding Computation Complete ===")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
