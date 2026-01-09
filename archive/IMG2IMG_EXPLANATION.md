# Image-to-Image Without Masking

## How This Works

This workflow uses **image-to-image (img2img)** generation instead of masking/inpainting. The model looks at the entire original image and regenerates it based on the prompt, preserving what it recognises while making the requested changes.

## Denoise Parameter (Critical!)

The **denoise** parameter controls how much the image changes:

- **0.3-0.4**: Very subtle changes, mostly preserves original
- **0.5-0.6**: Moderate changes, good balance
- **0.7-0.8**: Significant changes, may alter more than intended
- **0.9-1.0**: Almost complete regeneration (like text-to-image)

**For preserving the original person/pose:**
- Start with **0.3-0.5** for subtle clothing removal
- Use **0.5-0.7** if you need more change but still want to preserve the person

## How It Works

1. **LoadImage**: Loads your original image
2. **VAEEncode**: Converts image to latent space (the model's internal representation)
3. **KSampler**: Starts from this encoded image and denoises it based on your prompt
4. **VAEDecode**: Converts back to image
5. **SaveImage**: Saves result

The model uses its understanding of:
- What a "person" looks like
- What "clothing" is
- The pose and composition
- To regenerate the image without clothing while preserving the person

## Prompt Strategy

The prompt emphasises:
- **"exact same person"**: Preserve identity
- **"identical pose"**: Keep the pose
- **"identical background"**: Keep surroundings
- **"no clothing, bare skin"**: The desired change

## Limitations

- The model may not perfectly preserve the exact person if denoise is too high
- It relies on the model's "understanding" rather than pixel-perfect masking
- Results may vary significantly between runs (even with same seed) due to the stochastic nature

## Testing Strategy

Test different denoise values to find the sweet spot:
- Too low (0.2-0.3): May not remove clothing effectively
- Too high (0.8-1.0): Generates completely new images
- Sweet spot (0.4-0.6): Modifies clothing while preserving person
