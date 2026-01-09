# Simplified Text-to-Image Workflow

## Changes Made

The workflow has been simplified from an inpainting/masking workflow to a basic **text-to-image** workflow.

### Removed Nodes
- **LoadImage** (node 2): No longer loads an input image
- **VAEEncode** (node 8): No longer encodes an existing image
- **SetLatentNoiseMask** (node 12): Masking functionality removed
- **SolidMask** (node 14): Mask generation removed

### Kept/Modified Nodes
- **CheckpointLoaderSimple** (node 3): Loads the model checkpoint
- **KSampler** (node 4): 
  - Changed `latent_image` input from VAEEncode to EmptyLatentImage
  - Changed `denoise` default from 0.5 to 1.0 (full generation for text-to-image)
- **CLIPTextEncode** (node 9): Positive prompt (updated to a general landscape prompt)
- **CLIPTextEncode** (node 10): Negative prompt (kept standard quality/artifacts negative)
- **VAEDecode** (node 6): Decodes latent to image
- **SaveImage** (node 7): Saves the final result

### Added Nodes
- **EmptyLatentImage** (node 5): Creates an empty 512x512 latent for text-to-image generation

## Workflow Flow

1. **EmptyLatentImage** → Creates empty latent (512×512)
2. **CheckpointLoaderSimple** → Loads model, CLIP, and VAE
3. **CLIPTextEncode** (positive) → Encodes positive prompt
4. **CLIPTextEncode** (negative) → Encodes negative prompt
5. **KSampler** → Generates image from noise using prompts
6. **VAEDecode** → Converts latent to image
7. **SaveImage** → Saves the result

## Parameter Testing

This simplified workflow allows you to test how different parameters affect image generation:

- **denoise** (0.8-1.0): Strength of generation (1.0 = full generation from noise)
- **cfg** (5.0-10.0): How closely to follow the prompt
- **steps** (20, 25, 30): Number of denoising steps
- **sampler_name**: Different sampling algorithms
- **scheduler**: Different noise schedules

## Purpose

This workflow is designed for:
1. **Parameter exploration**: Understanding how each parameter affects output
2. **Model training data**: Generating datasets with varied parameters for training a model to predict optimal parameter combinations
3. **Optimisation research**: Finding the best parameter combinations for specific goals
