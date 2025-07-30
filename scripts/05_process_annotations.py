#!/usr/bin/env python3
"""
05_process_annotations.py

Process human annotations from Google Sheets export to:
1. Calculate inter-rater reliability (Cohen's kappa)
2. Generate consensus labels using majority voting
3. Analyze annotation confidence scores

Expected CSV format:
- image_id, prompt, character, annotator_1_binary, annotator_1_confidence, 
  annotator_2_binary, annotator_2_confidence, annotator_3_binary, annotator_3_confidence
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Tuple
import itertools


def calculate_pairwise_kappa(annotations_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Cohen's kappa for each pair of annotators."""
    kappa_scores = {}
    
    # Get annotator pairs
    annotator_pairs = [
        ('annotator_1', 'annotator_2'),
        ('annotator_1', 'annotator_3'),
        ('annotator_2', 'annotator_3')
    ]
    
    for ann1, ann2 in annotator_pairs:
        col1 = f'{ann1}_binary'
        col2 = f'{ann2}_binary'
        
        if col1 in annotations_df.columns and col2 in annotations_df.columns:
            # Filter out any NaN values
            mask = annotations_df[[col1, col2]].notna().all(axis=1)
            if mask.sum() > 0:
                kappa = cohen_kappa_score(
                    annotations_df.loc[mask, col1],
                    annotations_df.loc[mask, col2]
                )
                kappa_scores[f'{ann1}_vs_{ann2}'] = kappa
    
    # Calculate average kappa
    if kappa_scores:
        kappa_scores['average'] = np.mean(list(kappa_scores.values()))
    
    return kappa_scores


def calculate_fleiss_kappa(annotations_df: pd.DataFrame) -> float:
    """Calculate Fleiss' kappa for multiple annotators."""
    # Get binary columns
    binary_cols = [col for col in annotations_df.columns if col.endswith('_binary')]
    
    if len(binary_cols) < 2:
        return np.nan
    
    # Convert to matrix format needed for Fleiss' kappa
    n_items = len(annotations_df)
    n_annotators = len(binary_cols)
    
    # Count agreements for each category (0 and 1)
    category_counts = np.zeros((n_items, 2))  # 2 categories: 0 and 1
    
    for i, row in annotations_df.iterrows():
        for col in binary_cols:
            if pd.notna(row[col]):
                category_counts[i, int(row[col])] += 1
    
    # Calculate Fleiss' kappa
    n = n_annotators
    N = n_items
    
    # Calculate P_j (proportion of all assignments to category j)
    P_j = category_counts.sum(axis=0) / (N * n)
    
    # Calculate P_i (extent of agreement for item i)
    P_i = (category_counts ** 2).sum(axis=1) - n
    P_i = P_i / (n * (n - 1))
    
    # Calculate P_bar (mean of P_i)
    P_bar = P_i.mean()
    
    # Calculate P_e (chance agreement)
    P_e = (P_j ** 2).sum()
    
    # Calculate Fleiss' kappa
    if P_e == 1:
        return 1.0 if P_bar == 1 else 0.0
    
    kappa = (P_bar - P_e) / (1 - P_e)
    
    return kappa


def compute_consensus_labels(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consensus labels using majority voting.
    Also calculate agreement level and average confidence.
    """
    binary_cols = [col for col in annotations_df.columns if col.endswith('_binary')]
    confidence_cols = [col for col in annotations_df.columns if col.endswith('_confidence')]
    
    consensus_data = []
    
    for idx, row in annotations_df.iterrows():
        # Get binary votes
        votes = []
        confidences = []
        
        for i, binary_col in enumerate(binary_cols):
            if pd.notna(row[binary_col]):
                votes.append(int(row[binary_col]))
                
                # Get corresponding confidence
                conf_col = confidence_cols[i] if i < len(confidence_cols) else None
                if conf_col and pd.notna(row[conf_col]):
                    confidences.append(row[conf_col])
        
        if len(votes) > 0:
            # Majority vote
            consensus = 1 if sum(votes) > len(votes) / 2 else 0
            
            # Agreement level
            agreement = sum(v == consensus for v in votes) / len(votes)
            
            # Average confidence
            avg_confidence = np.mean(confidences) if confidences else np.nan
            
            # Unanimous decision
            unanimous = all(v == votes[0] for v in votes) if votes else False
            
            consensus_data.append({
                'consensus_binary': consensus,
                'agreement_level': agreement,
                'avg_confidence': avg_confidence,
                'unanimous': unanimous,
                'n_annotators': len(votes)
            })
        else:
            consensus_data.append({
                'consensus_binary': np.nan,
                'agreement_level': np.nan,
                'avg_confidence': np.nan,
                'unanimous': False,
                'n_annotators': 0
            })
    
    # Combine with original data
    consensus_df = pd.concat([
        annotations_df[['image_id', 'prompt', 'character', 'model'] 
                      if 'model' in annotations_df.columns 
                      else ['image_id', 'prompt', 'character']],
        pd.DataFrame(consensus_data)
    ], axis=1)
    
    return consensus_df


def analyze_annotation_statistics(annotations_df: pd.DataFrame, consensus_df: pd.DataFrame) -> Dict:
    """Compute comprehensive annotation statistics."""
    stats = {}
    
    # Overall statistics
    stats['overall'] = {
        'total_images': len(annotations_df),
        'infringement_rate': consensus_df['consensus_binary'].mean() if 'consensus_binary' in consensus_df else np.nan,
        'average_agreement': consensus_df['agreement_level'].mean() if 'agreement_level' in consensus_df else np.nan,
        'unanimous_decisions': consensus_df['unanimous'].sum() if 'unanimous' in consensus_df else 0,
        'average_confidence': consensus_df['avg_confidence'].mean() if 'avg_confidence' in consensus_df else np.nan
    }
    
    # Inter-rater reliability
    stats['kappa_scores'] = calculate_pairwise_kappa(annotations_df)
    stats['fleiss_kappa'] = calculate_fleiss_kappa(annotations_df)
    
    # Per-character statistics
    if 'character' in consensus_df.columns:
        stats['by_character'] = {}
        for character in consensus_df['character'].unique():
            char_data = consensus_df[consensus_df['character'] == character]
            stats['by_character'][character] = {
                'count': len(char_data),
                'infringement_rate': char_data['consensus_binary'].mean(),
                'avg_agreement': char_data['agreement_level'].mean(),
                'avg_confidence': char_data['avg_confidence'].mean()
            }
    
    # Per-model statistics (if available)
    if 'model' in consensus_df.columns:
        stats['by_model'] = {}
        for model in consensus_df['model'].unique():
            model_data = consensus_df[consensus_df['model'] == model]
            stats['by_model'][model] = {
                'count': len(model_data),
                'infringement_rate': model_data['consensus_binary'].mean(),
                'avg_agreement': model_data['agreement_level'].mean(),
                'avg_confidence': model_data['avg_confidence'].mean()
            }
    
    # Confidence analysis
    confidence_cols = [col for col in annotations_df.columns if col.endswith('_confidence')]
    if confidence_cols:
        all_confidences = []
        for col in confidence_cols:
            all_confidences.extend(annotations_df[col].dropna().values)
        
        if all_confidences:
            stats['confidence_distribution'] = {
                '0': sum(c == 0 for c in all_confidences) / len(all_confidences),
                '1': sum(c == 1 for c in all_confidences) / len(all_confidences),
                '2': sum(c == 2 for c in all_confidences) / len(all_confidences),
                '3': sum(c == 3 for c in all_confidences) / len(all_confidences),
                'mean': np.mean(all_confidences),
                'std': np.std(all_confidences)
            }
    
    return stats


def save_annotation_report(stats: Dict, output_file: Path):
    """Save annotation statistics to a readable text file."""
    with open(output_file, 'w') as f:
        f.write("Human Annotation Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 30 + "\n")
        for key, value in stats['overall'].items():
            if isinstance(value, float):
                f.write(f"{key:25s}: {value:8.4f}\n")
            else:
                f.write(f"{key:25s}: {value}\n")
        f.write("\n")
        
        # Inter-rater reliability
        f.write("Inter-Rater Reliability (Cohen's Kappa):\n")
        f.write("-" * 30 + "\n")
        for pair, kappa in stats['kappa_scores'].items():
            f.write(f"{pair:25s}: {kappa:8.4f}\n")
        f.write(f"\nFleiss' Kappa (all raters): {stats['fleiss_kappa']:8.4f}\n\n")
        
        # Character statistics
        if 'by_character' in stats:
            f.write("Statistics by Character:\n")
            f.write("-" * 30 + "\n")
            for char, char_stats in sorted(stats['by_character'].items()):
                f.write(f"\n{char}:\n")
                for key, value in char_stats.items():
                    if isinstance(value, float):
                        f.write(f"  {key:20s}: {value:8.4f}\n")
                    else:
                        f.write(f"  {key:20s}: {value}\n")
        
        # Model statistics
        if 'by_model' in stats:
            f.write("\nStatistics by Model:\n")
            f.write("-" * 30 + "\n")
            for model, model_stats in stats['by_model'].items():
                f.write(f"\n{model}:\n")
                for key, value in model_stats.items():
                    if isinstance(value, float):
                        f.write(f"  {key:20s}: {value:8.4f}\n")
                    else:
                        f.write(f"  {key:20s}: {value}\n")
        
        # Confidence distribution
        if 'confidence_distribution' in stats:
            f.write("\nConfidence Score Distribution:\n")
            f.write("-" * 30 + "\n")
            conf_dist = stats['confidence_distribution']
            for score in ['0', '1', '2', '3']:
                if score in conf_dist:
                    f.write(f"Score {score}: {conf_dist[score]:8.2%}\n")
            f.write(f"\nMean confidence: {conf_dist['mean']:8.4f}\n")
            f.write(f"Std deviation:   {conf_dist['std']:8.4f}\n")


def merge_with_metadata(consensus_df: pd.DataFrame, metadata_dir: Path) -> pd.DataFrame:
    """Merge consensus labels with image metadata."""
    # Try to load metadata from image generation
    all_metadata = []
    
    for model_dir in metadata_dir.iterdir():
        if model_dir.is_dir():
            metadata_file = model_dir / 'metadata.json'
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    for item in metadata:
                        item['model'] = model_dir.name
                    all_metadata.extend(metadata)
    
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        # Merge on image_id
        if 'image_id' in consensus_df.columns and 'image_id' in metadata_df.columns:
            merged_df = consensus_df.merge(
                metadata_df[['image_id', 'model', 'prompt_type', 'prompt_label']], 
                on='image_id', 
                how='left'
            )
            return merged_df
    
    return consensus_df


def main():
    parser = argparse.ArgumentParser(description='Process human annotations')
    parser.add_argument('--annotations_csv', type=str, required=True,
                        help='CSV file with human annotations')
    parser.add_argument('--output_dir', type=str, default='data/annotations',
                        help='Output directory for processed annotations')
    parser.add_argument('--image_metadata_dir', type=str, default='data/images',
                        help='Directory containing image metadata')
    parser.add_argument('--models', nargs='+',
                        default=['playground', 'sdxl', 'deepfloyd'],
                        help='Model names for splitting annotations')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print("Loading annotations...")
    annotations_df = pd.read_csv(args.annotations_csv)
    print(f"Loaded {len(annotations_df)} annotations")
    
    # Calculate inter-rater reliability
    print("\nCalculating inter-rater reliability...")
    kappa_scores = calculate_pairwise_kappa(annotations_df)
    fleiss_kappa = calculate_fleiss_kappa(annotations_df)
    
    print("Cohen's Kappa scores:")
    for pair, kappa in kappa_scores.items():
        print(f"  {pair}: {kappa:.4f}")
    print(f"Fleiss' Kappa: {fleiss_kappa:.4f}")
    
    # Compute consensus labels
    print("\nComputing consensus labels...")
    consensus_df = compute_consensus_labels(annotations_df)
    
    # Try to merge with metadata
    if Path(args.image_metadata_dir).exists():
        print("Merging with image metadata...")
        consensus_df = merge_with_metadata(consensus_df, Path(args.image_metadata_dir))
    
    # Save consensus labels
    consensus_file = output_dir / 'consensus_labels.csv'
    consensus_df.to_csv(consensus_file, index=False)
    print(f"Saved consensus labels to {consensus_file}")
    
    # Analyze statistics
    print("\nAnalyzing annotation statistics...")
    stats = analyze_annotation_statistics(annotations_df, consensus_df)
    
    # Save statistics report
    report_file = output_dir / 'annotation_statistics.txt'
    save_annotation_report(stats, report_file)
    print(f"Saved statistics report to {report_file}")
    
    # Save per-model annotations if model column exists
    if 'model' in consensus_df.columns:
        for model in args.models:
            model_data = consensus_df[consensus_df['model'] == model]
            if len(model_data) > 0:
                model_file = output_dir / f'{model}_annotations.csv'
                model_data.to_csv(model_file, index=False)
                print(f"Saved {model} annotations to {model_file}")
                
                # Calculate model-specific kappa if original annotations have model info
                if 'model' in annotations_df.columns:
                    model_annotations = annotations_df[annotations_df['model'] == model]
                    if len(model_annotations) > 0:
                        model_kappa = calculate_pairwise_kappa(model_annotations)
                        print(f"\n{model} Cohen's Kappa:")
                        for pair, kappa in model_kappa.items():
                            print(f"  {pair}: {kappa:.4f}")
    
    # Create a simplified output for quadrant analysis
    if 'consensus_binary' in consensus_df.columns:
        quadrant_df = consensus_df[[
            'image_id', 'character', 'consensus_binary', 'agreement_level', 'avg_confidence'
        ]].copy()
        
        # Rename for compatibility with quadrant analysis
        quadrant_df.rename(columns={
            'consensus_binary': 'is_infringing',
            'character': 'target'
        }, inplace=True)
        
        quadrant_file = output_dir / 'annotations_for_quadrant.csv'
        quadrant_df.to_csv(quadrant_file, index=False)
        print(f"\nSaved quadrant analysis format to {quadrant_file}")
    
    print("\n=== Annotation Processing Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"\nSummary:")
    print(f"  Total images: {stats['overall']['total_images']}")
    print(f"  Infringement rate: {stats['overall']['infringement_rate']:.2%}")
    print(f"  Average agreement: {stats['overall']['average_agreement']:.2%}")
    print(f"  Average confidence: {stats['overall']['average_confidence']:.2f}")


if __name__ == '__main__':
    main()
