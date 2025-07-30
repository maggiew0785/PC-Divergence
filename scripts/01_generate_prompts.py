#!/usr/bin/env python3
"""
01_generate_prompts.py

Generate prompts for text-to-image experiments using two methodologies:
1. Natural language prompts via GPT-o3 (specificity gradient)
2. Keyword-based prompts from COPYCAT framework (5/10/15/20 keywords)

Total: 20 prompts per character (10 natural + 10 keyword)
"""

import os
import re
import argparse
import pandas as pd
import random
from pathlib import Path
from openai import OpenAI
from typing import List, Dict


# Character list (Kirby removed due to poor results)
CHARACTERS = [
    "Mario",
    "Mickey Mouse", 
    "Yoda",
    "Pikachu",
    "Spider-man",
    "Batman",
    "Tinkerbell",
    "Spongebob Squarepants",
    "Elsa",
    "Sonic the Hedgehog"
]


def sanitize_filename(name: str) -> str:
    """Convert character name to safe filename."""
    return re.sub(r'[^\w\s]', '_', name).replace(' ', '_')


def generate_natural_language_prompts(character: str, client: OpenAI) -> List[str]:
    """
    Generate 10 natural language prompts with increasing specificity gradient.
    Uses GPT-o3 to create prompts from general to specific descriptions.
    """
    prompt_template = f"""
    You are helping with an experiment on text-to-image models and copyrighted character recognition.

    Given the character name: {character}

    Generate 10 distinct image generation prompts that exhibit a gradient of specificity:
    
    1-3: Begin with general descriptions that include subtle hints (color, theme, style) without revealing full identity
    4-6: Include core identifying visual features (for Mario: red cap, mustache, overalls, plumber, etc.)
    7-9: Add more contextual elements, accessories, environmental details that reinforce identity
    10: State the character's full name with comprehensive descriptive details

    Each subsequent prompt should be more detailed and longer than the previous.
    Output the prompts as a simple numbered list with no additional commentary.
    """
    
    try:
        completion = client.chat.completions.create(
            model="o3-mini-2025-01-31",
            messages=[{"role": "user", "content": prompt_template}]
        )
        
        # Extract and clean the prompts
        response = completion.choices[0].message.content
        prompts = []
        
        for line in response.strip().split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                # Remove the number prefix
                prompt_text = re.sub(r'^\d+\.\s*', '', line.strip())
                prompts.append(prompt_text)
        
        # Ensure we have exactly 10 prompts
        if len(prompts) != 10:
            print(f"Warning: Generated {len(prompts)} prompts for {character}, expected 10")
        
        return prompts[:10]  # Return max 10 prompts
        
    except Exception as e:
        print(f"Error generating prompts for {character}: {e}")
        return []


def generate_keyword_prompts(character: str, keywords_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Generate keyword-based prompts using COPYCAT framework.
    Creates prompts with 5, 10, 15, and 20 keywords.
    
    Returns dict with keys: '5_keywords_1', '5_keywords_2', '5_keywords_3',
                           '10_keywords_1', '10_keywords_2', '10_keywords_3',
                           '15_keywords_1', '15_keywords_2', '15_keywords_3',
                           '20_keywords'
    """
    # Find keywords for this character
    char_keywords = keywords_df[keywords_df['target'] == character]
    
    if char_keywords.empty:
        print(f"Warning: No keywords found for {character}")
        return {}
    
    # Extract keywords list
    keywords_str = char_keywords.iloc[0]['prompt']
    keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
    
    if len(keywords) < 20:
        print(f"Warning: Only {len(keywords)} keywords for {character}, expected 20+")
    
    prompts = {}
    
    # Generate 20-keyword prompt (all keywords)
    prompts['20_keywords'] = ', '.join(keywords[:20])
    
    # Generate multiple samples for 5, 10, 15 keywords
    for n_keywords in [5, 10, 15]:
        for i in range(1, 4):  # 3 variations each
            if len(keywords) >= n_keywords:
                sampled = random.sample(keywords[:20], n_keywords)
                prompts[f'{n_keywords}_keywords_{i}'] = ', '.join(sampled)
    
    return prompts


def save_prompts_to_csv(all_prompts: Dict, output_path: Path):
    """Save all prompts to a structured CSV file."""
    rows = []
    
    for character, prompt_data in all_prompts.items():
        # Natural language prompts
        for i, prompt in enumerate(prompt_data['natural_language'], 1):
            rows.append({
                'target': character,
                'prompt_type': 'natural_language',
                'prompt_label': f'natural_{i}',
                'prompt': prompt
            })
        
        # Keyword prompts
        for label, prompt in prompt_data['keywords'].items():
            rows.append({
                'target': character,
                'prompt_type': 'keywords',
                'prompt_label': label,
                'prompt': prompt
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} prompts to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate prompts for PC-D experiments')
    parser.add_argument('--openai_key', type=str, required=True,
                        help='OpenAI API key for GPT-o3')
    parser.add_argument('--keywords_csv', type=str, required=True,
                        help='Path to LAION2B keywords CSV')
    parser.add_argument('--output_dir', type=str, default='data/prompts',
                        help='Output directory for prompts')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=args.openai_key)
    
    # Load keywords
    keywords_df = pd.read_csv(args.keywords_csv)
    
    # Generate prompts for each character
    all_prompts = {}
    
    for character in CHARACTERS:
        print(f"\nGenerating prompts for: {character}")
        
        # Generate natural language prompts
        print("  Generating natural language prompts...")
        nl_prompts = generate_natural_language_prompts(character, client)
        
        # Generate keyword prompts
        print("  Generating keyword prompts...")
        kw_prompts = generate_keyword_prompts(character, keywords_df)
        
        all_prompts[character] = {
            'natural_language': nl_prompts,
            'keywords': kw_prompts
        }
        
        # Also save individual character files for reference
        char_dir = os.path.join(args.output_dir, 'by_character', sanitize_filename(character))
        os.makedirs(char_dir, exist_ok=True)
        
        # Save natural language prompts
        with open(os.path.join(char_dir, 'natural_language.txt'), 'w') as f:
            for i, prompt in enumerate(nl_prompts, 1):
                f.write(f"{i}. {prompt}\n")
        
        # Save keyword prompts
        with open(os.path.join(char_dir, 'keywords.txt'), 'w') as f:
            for label, prompt in kw_prompts.items():
                f.write(f"{label}: {prompt}\n")
    
    # Save combined CSV
    output_csv = os.path.join(args.output_dir, 'all_prompts.csv')
    save_prompts_to_csv(all_prompts, Path(output_csv))
    
    # Print summary
    print("\n=== Prompt Generation Summary ===")
    print(f"Characters processed: {len(CHARACTERS)}")
    print(f"Prompts per character: 20 (10 natural + 10 keyword)")
    print(f"Total prompts generated: {len(CHARACTERS) * 20}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()