#!/usr/bin/env python3
"""
06_quadrant_analysis.py

Compute the 4-quadrant responsibility compass analysis for PC-D scores.
This is the core contribution of the paper - analyzing how generated images
distribute across different responsibility zones.

Quadrants:
- Q1: High PC-D, Low Prompt Specificity (Model-Driven Risk)
- Q2: High PC-D, High Prompt Specificity (Mixed Responsibility)
- Q3: Low PC-D, Low Prompt Specificity (Safe Zone)
- Q4: Low PC-D, High Prompt Specificity (User-Driven)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def load_data(model_name, annotations_path, pcd_path):
    """Load and merge annotation data with PC-D scores."""
    # Load annotations with prompt specificity
    annotations = pd.read_csv(annotations_path)
    
    # Load PC-D scores
    pcd_scores = pd.read_csv(pcd_path)
    
    # Merge on common identifiers (adjust column names as needed)
    # Assuming both have 'target' and can be matched by row order
    if len(annotations) != len(pcd_scores):
        print(f"Warning: Annotation rows ({len(annotations)}) != PC-D rows ({len(pcd_scores)})")
    
    # Add PC-D scores to annotations
    annotations['pc_d'] = pcd_scores['similarity_difference'].values
    annotations['model'] = model_name
    
    # Rename for clarity
    annotations['prompt_specificity'] = annotations['openclip_similarity']
    annotations['is_infringing'] = annotations['human_annotator_1']
    
    return annotations


def compute_quadrant_thresholds(all_data):
    """
    Compute unified thresholds using global medians:
    1. PC-D threshold: median of all PC-D scores across all data
    2. Prompt specificity threshold: median of all prompt specificity scores
    """
    # Use global medians instead of F1-optimized thresholds
    pc_d_threshold = all_data['pc_d'].median()
    prompt_spec_threshold = all_data['prompt_specificity'].median()
    
    return pc_d_threshold, prompt_spec_threshold


def assign_quadrants(data, pc_d_threshold, prompt_spec_threshold):
    """Assign each image to a quadrant based on thresholds."""
    quadrants = []
    
    for _, row in data.iterrows():
        high_pcd = row['pc_d'] >= pc_d_threshold
        high_prompt = row['prompt_specificity'] >= prompt_spec_threshold
        
        if high_pcd and not high_prompt:
            quadrant = 'Q1'  # Model-Driven Risk
        elif high_pcd and high_prompt:
            quadrant = 'Q2'  # Mixed Responsibility
        elif not high_pcd and not high_prompt:
            quadrant = 'Q3'  # Safe Zone
        else:  # not high_pcd and high_prompt
            quadrant = 'Q4'  # User-Driven
        
        quadrants.append(quadrant)
    
    data['quadrant'] = quadrants
    return data


def calculate_quadrant_statistics(data):
    """Calculate infringement rates and counts per quadrant."""
    stats = []
    
    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = data[data['quadrant'] == quadrant]
        
        if len(q_data) > 0:
            infringement_rate = q_data['is_infringing'].mean() * 100
            count = len(q_data)
            
            stats.append({
                'quadrant': quadrant,
                'count': count,
                'infringement_rate': infringement_rate,
                'infringing_count': q_data['is_infringing'].sum()
            })
        else:
            # Handle empty quadrants
            stats.append({
                'quadrant': quadrant,
                'count': 0,
                'infringement_rate': 0.0,
                'infringing_count': 0
            })
    
    return pd.DataFrame(stats)


def plot_quadrant_scatter(data, pc_d_threshold, prompt_spec_threshold, output_path):
    """Generate Figure 7: Quadrant scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define colors for infringing vs non-infringing
    colors = ['blue' if x == 0 else 'red' for x in data['is_infringing']]
    
    # Scatter plot
    scatter = ax.scatter(
        data['prompt_specificity'],
        data['pc_d'],
        c=colors,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add threshold lines
    ax.axhline(y=pc_d_threshold, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=prompt_spec_threshold, color='black', linestyle='--', linewidth=1.5)
    
    # Add quadrant labels with statistics
    stats = calculate_quadrant_statistics(data)
    
    # Q1: Top-left
    q1_stats = stats[stats['quadrant'] == 'Q1'].iloc[0]
    ax.text(0.1, 0.9, f"Q1: {q1_stats['infringement_rate']:.1f}% inf. ({q1_stats['count']})",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    # Q2: Top-right
    q2_stats = stats[stats['quadrant'] == 'Q2'].iloc[0]
    ax.text(0.9, 0.9, f"Q2: {q2_stats['infringement_rate']:.1f}% inf. ({q2_stats['count']})",
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # Q3: Bottom-left
    q3_stats = stats[stats['quadrant'] == 'Q3'].iloc[0]
    ax.text(0.1, 0.1, f"Q3: {q3_stats['infringement_rate']:.1f}% inf. ({q3_stats['count']})",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Q4: Bottom-right
    q4_stats = stats[stats['quadrant'] == 'Q4'].iloc[0]
    ax.text(0.9, 0.1, f"Q4: {q4_stats['infringement_rate']:.1f}% inf. ({q4_stats['count']})",
            transform=ax.transAxes, fontsize=10, horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Labels and title
    ax.set_xlabel('Prompt Specificity', fontsize=12)
    ax.set_ylabel('PC-D Score', fontsize=12)
    ax.set_title('Quadrant Analysis of PC-D Scores', fontsize=14)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Non-infringing'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Infringing')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Compute quadrant analysis for PC-D scores using global medians')
    parser.add_argument('--annotations_dir', type=str, required=True,
                        help='Directory containing annotation CSV files')
    parser.add_argument('--pcd_dir', type=str, required=True,
                        help='Directory containing PC-D score CSV files')
    parser.add_argument('--output_dir', type=str, default='results/quadrant_analysis',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model names
    models = {
        'playground': 'Playground-v2.5',
        'sdxl': 'SDXL',
        'deepfloyd': 'DeepFloyd-IF'
    }
    
    # Load all data
    all_data = []
    for model_key, model_name in models.items():
        print(f"\nProcessing {model_name}...")
        
        # Find annotation and PC-D files
        annotation_file = os.path.join(args.annotations_dir, f'{model_key}_annotations.csv')
        pcd_file = os.path.join(args.pcd_dir, f'pcd_{model_key}.csv')
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found: {annotation_file}")
            continue
        if not os.path.exists(pcd_file):
            print(f"Warning: PC-D file not found: {pcd_file}")
            continue
        
        # Load and merge data
        model_data = load_data(model_name, annotation_file, pcd_file)
        all_data.append(model_data)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Compute unified thresholds using global medians
    print("\nComputing unified thresholds using global medians...")
    pc_d_threshold, prompt_spec_threshold = compute_quadrant_thresholds(combined_data)
    print(f"PC-D threshold (global median): {pc_d_threshold:.4f}")
    print(f"Prompt specificity threshold (global median): {prompt_spec_threshold:.4f}")
    
    # Assign quadrants
    combined_data = assign_quadrants(combined_data, pc_d_threshold, prompt_spec_threshold)
    
    # Calculate overall statistics
    print("\nOverall Quadrant Statistics:")
    overall_stats = calculate_quadrant_statistics(combined_data)
    print(overall_stats)
    
    # Save results
    combined_data.to_csv(os.path.join(args.output_dir, 'quadrant_assignments.csv'), index=False)
    overall_stats.to_csv(os.path.join(args.output_dir, 'quadrant_statistics.csv'), index=False)
    
    # Generate plots for each model
    for model in combined_data['model'].unique():
        model_data = combined_data[combined_data['model'] == model]
        model_key = model.lower().replace('-', '_').replace(' ', '_')
        
        # Plot quadrant scatter
        plot_path = os.path.join(args.output_dir, f'figure_7_{model_key}.png')
        plot_quadrant_scatter(model_data, pc_d_threshold, prompt_spec_threshold, plot_path)
        print(f"Saved plot: {plot_path}")
        
        # Calculate model-specific statistics
        model_stats = calculate_quadrant_statistics(model_data)
        model_stats.to_csv(
            os.path.join(args.output_dir, f'quadrant_statistics_{model_key}.csv'),
            index=False
        )
    
    # Also create combined plot
    plot_path = os.path.join(args.output_dir, 'figure_7_combined.png')
    plot_quadrant_scatter(combined_data, pc_d_threshold, prompt_spec_threshold, plot_path)
    print(f"Saved combined plot: {plot_path}")
    
    print("\nQuadrant analysis complete!")


if __name__ == '__main__':
    main()
