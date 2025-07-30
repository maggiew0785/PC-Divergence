# PC-D Evaluation Code (Anonymous Submission)

This repository contains code to compute **Prompt-Character Divergence (PC-D)** scores for evaluating semantic drift in text-to-image models. The code supports generation-free evaluation of model outputs and is designed for reproducibility on standard hardware.

This code was submitted as supplementary material to a NeurIPS 2025 paper submission.

## Overview

PC-Divergence (PC-D) is defined as:
```
PC-D = cos(image, character) - cos(image, prompt)
```

Where:
- `cos(image, character)` measures similarity between generated image and character name
- `cos(image, prompt)` measures similarity between generated image and input prompt
- Higher PC-D scores indicate the model generated character-specific features beyond what was prompted

## Project Structure

```
pc-divergence/
├── data/
│   ├── characters.txt              # List of copyrighted characters studied
│   └── annotations/                # PC-D scores and human annotations per model
│       ├── sdxl.csv
│       ├── playground-v2.5.csv
│       └── deepfloyd-if.csv
├── scripts/
│   ├── 01_generate_prompts.py      # Generate prompts using GPT-o3 and COPYCAT
│   ├── 02_generate_images.py       # Generate images with SDXL, DeepFloyd, Playground
│   ├── 03_compute_embeddings.py    # Compute OpenCLIP embeddings
│   ├── 04_compute_pcd.py           # Calculate PC-D scores
│   ├── 05_process_annotations.py   # Process human annotations
│   └── 06_quadrant_analysis.py     # 4-quadrant responsibility analysis
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
# For reviewers only: use local folder after unzipping the anonymous .zip
cd pc-divergence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Key libraries used:
* `torch`
* `open_clip_torch`
* `sentence-transformers`
* `transformers`
* `pandas`, `numpy`, `scipy`, `tqdm`
* `ftfy` (for text normalization)
See `requirements.txt` for the full list.
```

### 2. Generate Prompts

Create prompts of gradient specificity using two methodologies:
- **Natural Language**: [Optional] This script requires an OpenAI API key and was used for prompt generation. Prompts are already provided in `data/prompts/`. (10 per character)
- **Keywords**: Random sample 5/10/15/20 co-occurence keywords (10 per character) from the LAION-2B dataset (following a co-occurrence keyword strategy inspired by prior frameworks)

```bash
python scripts/01_generate_prompts.py \
    --openai_key YOUR_OPENAI_API_KEY \
    --keywords_csv path/to/laion2b_keywords.csv \
    --output_dir data/prompts
```

### 3. Generate Images

Generate images using three open-source models:

```bash
python scripts/02_generate_images.py \
    --prompts_csv data/prompts/all_prompts.csv \
    --output_dir data/images \
    --models playground sdxl deepfloyd
```

**Models used:**
- **Stable Diffusion XL (SDXL)**: `stabilityai/stable-diffusion-xl-base-1.0`
- **DeepFloyd IF**: Two-stage generation with `DeepFloyd/IF-I-XL-v1.0` and `DeepFloyd/IF-II-L-v1.0`
- **Playground-v2.5**: `playgroundai/playground-v2.5-1024px-aesthetic`

### 4. Compute Embeddings

Calculate OpenCLIP embeddings for prompts, character names, and generated images:

```bash
python scripts/03_compute_embeddings.py \
    --prompts_csv data/prompts/all_prompts.csv \
    --characters_file data/characters.txt \
    --image_dir data/images \
    --output_dir results/embeddings
```

### 5. Calculate PC-D Scores

Compute PC-D scores from the embeddings:

```bash
python scripts/04_compute_pcd.py \
    --embeddings_dir results/embeddings \
    --output_dir results/pcd_scores
```

### 6. Process Human Annotations

Process human annotations to compute inter-rater reliability and consensus labels:

```bash
python scripts/05_process_annotations.py \
    --annotations_csv data/annotations.csv \
    --output_dir data/annotations \
    --image_metadata_dir data/images
```

### 7. Quadrant Analysis

Generate the 4-quadrant responsibility compass analysis:

```bash
python scripts/06_quadrant_analysis.py \
    --annotations_dir data/annotations \
    --pcd_dir results/pcd_scores \
    --output_dir results/quadrant_analysis
```

## Methodology

### Prompt Generation

**Natural Language Prompts**: Generated using GPT-o3 with a specificity gradient:
1. Prompts 1-3: General descriptions with subtle hints
2. Prompts 4-6: Core identifying visual features
3. Prompts 7-9: Additional contextual elements
4. Prompt 10: Full character name with comprehensive details

**Keyword Prompts**: Based on COPYCAT framework using LAION-2B keywords:
- 3 variations each of 5, 10, 15 keywords
- 1 prompt with all 20 keywords

### Characters Studied

The project focuses on 10 popular copyrighted characters found in `data/characters.txt`

### Quadrant Analysis

The responsibility compass divides results into four quadrants:

- **Q1 (Model-Driven Risk)**: High PC-D, Low Prompt Specificity
- **Q2 (Mixed Responsibility)**: High PC-D, High Prompt Specificity  
- **Q3 (Safe Zone)**: Low PC-D, Low Prompt Specificity
- **Q4 (User-Driven)**: Low PC-D, High Prompt Specificity

## Results

Results are saved in structured CSV files:
- `results/embeddings/all_embeddings.csv`: All computed embeddings
- `results/pcd_scores/pcd_all_models.csv`: PC-D scores for all images
- `results/quadrant_analysis/quadrant_assignments.csv`: Quadrant assignments
- `data/annotations/consensus_labels.csv`: Human annotation consensus

## Dependencies

Key dependencies:
- **PyTorch**: Deep learning framework
- **OpenCLIP**: For computing embeddings
- **Diffusers**: HuggingFace diffusion models
- **OpenAI**: GPT-o3 API for prompt generation
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Inter-rater reliability metrics

See `requirements.txt` for complete list with version requirements.

## Hardware Requirements

- **GPU**: CUDA-compatible GPU with ≥8GB VRAM recommended for image generation
- **RAM**: ≥16GB system RAM
- **Storage**: ~50GB for generated images and embeddings

## 🔐 Anonymity Statement

This codebase has been anonymized for double-blind reviewing. No author-identifying metadata or URLs are included.

Please contact the authors after acceptance for de-anonymized code, models, or datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
