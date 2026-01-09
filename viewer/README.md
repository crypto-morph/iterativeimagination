# Iteration Viewer

A web-based viewer for visualizing iteration images and metadata from Iterative Imagination projects.

## Features

- 📊 Browse all projects and runs
- 🖼️ View iteration images in a clean grid layout
- 📈 See scores, parameters, and prompts for each iteration
- ✅ Track acceptance criteria progress
- 🔍 Compare iterations side-by-side
- 📱 Responsive design

## Usage

1. **Install dependencies** (if not already installed):
   ```bash
   pip install Flask PyYAML
   ```

2. **Start the viewer**:
   ```bash
   cd viewer
   python app.py
   ```

3. **Open in browser**:
   Navigate to `http://localhost:5000`

## Navigation

- **Home page**: Lists all projects with their runs
- **Run page**: Shows all iterations for a specific run with:
  - Original input image (toggleable)
  - Iteration images with metadata panels
  - Scores, parameters, prompts, and criteria results
  - Differences detected between iterations

## Controls

- **Show original image**: Toggle visibility of the input image
- **Show metadata panels**: Toggle detailed metadata for each iteration

## Project Structure

The viewer expects projects in the standard structure:
```
projects/
  <project_name>/
    working/
      <run_id>/
        images/
          iteration_*.png
        metadata/
          iteration_*_metadata.json
    input/
      input.png
```
