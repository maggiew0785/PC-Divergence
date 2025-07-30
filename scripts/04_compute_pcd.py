#!/usr/bin/env python3
"""
04_compute_pcd.py

Calculate Prompt-Character Divergence (PC-D) scores from pre-computed embeddings.
PC-D = cos(image, character) - cos(image, prompt)

This script takes the embeddings computed by 03_compute_embeddings.py and
calculates the PC-D score for each generated image.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def compute_pcd_scores(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PC-D scores from embedding similarities.
    
    PC-D = cos(image, character) - cos(image, prompt)
    """
    # Calculate PC-D
    embeddings_df['pc_d'] = embeddings_df['cos_img_char'] - embeddings_df['cos_img_prompt']
    
    return embeddings_df


def analyze_pcd_distribution(pcd_df: pd.DataFrame) -> Dict:
    """Compute summary statistics for PC-D scores."""
    stats = {
        'overall': {
            'mean': pcd_df['pc_d'].mean(),
            'std': pcd_df['pc_d'].std(),
            'median': pcd_df['pc_d'].median(),
            'min': pcd_df['pc_d'].min(),
            'max': pcd_df['pc_d'].max(),
            'q1': pcd_df['pc_d'].quantile(0.25),
            'q3': pcd_df['pc_d'].quantile(0.75)
        }
    }
    
    # Per-model statistics
    stats['by_model'] = {}
    for model in pcd_df['model'].unique():
        model_data = pcd_df[pcd_df['model'] == model]['pc_d']
        stats['by_model'][model] = {
            'mean': model_data.mean(),
            'std': model_data.std(),
            'median': model_data.median(),
            'count': len(model_data)
        }
    
    # Per-character statistics
    stats['by_character'] = {}
    for character in pcd_df['character'].unique():
        char_data = pcd_df[pcd_df['character'] == character]['pc_d']
        stats['by_character'][character] = {
            'mean': char_data.mean(),
            'std': char_data.std(),
            'median': char_data.median(),
            'count': len(char_data)
        }
    
    # Per prompt type statistics
    stats['by_prompt_type'] = {}
    for ptype in pcd_df['prompt_type'].unique():
        type_data = pcd_df[pcd_df['prompt_type'] == ptype]['pc_d']
        stats['by_prompt_type'][ptype] = {
            'mean': type_data.mean(),
            'std': type_data.std(),
            'median': type_data.median(),
            'count': len(type_data)
        }
    
    return stats


def save_statistics_report(stats: Dict, output_file: Path):
    """Save PC-D statistics to a readable text file."""
    with open(output_file, 'w') as f:
        f.write("PC-D Score Statistics Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 30 + "\n")
        for key, value in stats['overall'].items():
            f.write(f"{key:10s}: {value:8.4f}\n")
        f.write("\n")
        
        # Model statistics
        f.write("Statistics by Model:\n")
        f.write("-" * 30 + "\n")
        for model, model_stats in stats['by_model'].items():
            f.write(f"\n{model}:\n")
            for key, value in model_stats.items():
                if key == 'count':
                    f.write(f"  {key:8s}: {value:8d}\n")
                else:
                    f.write(f"  {key:8s}: {value:8.4f}\n")
        f.write("\n")
        
        # Character statistics
        f.write("Statistics by Character:\n")
        f.write("-" * 30 + "\n")
        for char, char_stats in sorted(stats['by_character'].items()):
            f.write(f"\n{char}:\n")
            for key, value in char_stats.items():
                if key == 'count':
                    f.write(f"  {key:8s}: {value:8d}\n")
                else:
                    f.write(f"  {key:8s}: {value:8.4f}\n")
        f.write("\n")
        
        # Prompt type statistics
        f.write("Statistics by Prompt Type:\n")
        f.write("-" * 30 + "\n")
        for ptype, type_stats in stats['by_prompt_type'].items():
            f.write(f"\n{ptype}:\n")
            for key, value in type_stats.items():
                if key == 'count':
                    f.write(f"  {key:8s}: {value:8d}\n")
                else:
                    f.write(f"  {key:8s}: {value:8.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Compute PC-D scores from embeddings')
    parser.add_argument('--embeddings_dir', type=str, default='results/embeddings',
                        help='Directory containing embedding CSV files')
    parser.add_argument('--output_dir', type=str, default='results/pcd_scores',
                        help='Output directory for PC-D scores')
    parser.add_argument('--models', nargs='+', 
                        default=['playground', 'sdxl', 'deepfloyd'],
                        help='Models to process')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process combined embeddings if available
    combined_file = Path(args.embeddings_dir) / 'all_embeddings.csv'
    if combined_file.exists():
        print("Processing combined embeddings file...")
        embeddings_df = pd.read_csv(combined_file)
        
        # Compute PC-D scores
        pcd_df = compute_pcd_scores(embeddings_df)
        
        # Save results
        output_file = output_dir / 'pcd_all_models.csv'
        pcd_df.to_csv(output_file, index=False)
        print(f"Saved PC-D scores to {output_file}")
        
        # Compute and save statistics
        stats = analyze_pcd_distribution(pcd_df)
        stats_file = output_dir / 'pcd_statistics.txt'
        save_statistics_report(stats, stats_file)
        print(f"Saved statistics to {stats_file}")
        
    # Process individual model files
    for model_name in args.models:
        embeddings_file = Path(args.embeddings_dir) / f'embeddings_{model_name}.csv'
        
        if not embeddings_file.exists():
            print(f"Warning: Embeddings file not found for {model_name}")
            continue
        
        print(f"\nProcessing {model_name} embeddings...")
        
        # Load embeddings
        embeddings_df = pd.read_csv(embeddings_file)
        
        # Compute PC-D scores
        pcd_df = compute_pcd_scores(embeddings_df)
        
        # Save results
        output_file = output_dir / f'pcd_{model_name}.csv'
        pcd_df.to_csv(output_file, index=False)
        print(f"Saved PC-D scores to {output_file}")
        
        # Save model-specific statistics
        model_stats = analyze_pcd_distribution(pcd_df)
        model_stats_file = output_dir / f'pcd_statistics_{model_name}.txt'
        save_statistics_report(model_stats, model_stats_file)
    
    # Create summary CSV with key columns for downstream analysis
    if combined_file.exists():
        summary_df = pcd_df[['model', 'character', 'prompt_label', 'prompt_type', 
                            'prompt_char_similarity', 'cos_img_prompt', 
                            'cos_img_char', 'pc_d']].copy()
        
        # Rename prompt_char_similarity to prompt_specificity for clarity
        summary_df.rename(columns={'prompt_char_similarity': 'prompt_specificity'}, inplace=True)
        
        summary_file = output_dir / 'pcd_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary for quadrant analysis to {summary_file}")
    
    print("\n=== PC-D Computation Complete ===")
    print(f"Output directory: {output_dir}")
    
    # Print summary statistics
    if combined_file.exists():
        print("\nOverall PC-D Statistics:")
        print(f"  Mean: {stats['overall']['mean']:.4f}")
        print(f"  Median: {stats['overall']['median']:.4f}")
        print(f"  Std Dev: {stats['overall']['std']:.4f}")
        print(f"  Range: [{stats['overall']['min']:.4f}, {stats['overall']['max']:.4f}]")


if __name__ == '__main__':
    main()
