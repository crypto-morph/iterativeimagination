#!/usr/bin/env python3
"""
ComfyUI Outputs Gallery - Simple web server to browse, delete, and keep images
"""

import os
import shutil
import argparse
from pathlib import Path
from flask import Flask, render_template_string, send_from_directory, request, jsonify

app = Flask(__name__)

# Configuration
OUTPUTS_DIR = Path.home() / "ComfyUI" / "output"
KEEP_DIR = OUTPUTS_DIR / "keep"
TRASH_DIR = OUTPUTS_DIR / "trash"

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}


def get_images_from_folder(folder_path):
    """Get list of image files from a specific folder."""
    if not folder_path.exists():
        return []
    
    images = []
    for file in sorted(folder_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append({
                'name': file.name,
                'size': file.stat().st_size,
                'mtime': file.stat().st_mtime
            })
    return images


def get_images(view='main'):
    """Get list of image files based on view type."""
    if view == 'keep':
        return get_images_from_folder(KEEP_DIR)
    elif view == 'trash':
        return get_images_from_folder(TRASH_DIR)
    else:  # main
        if not OUTPUTS_DIR.exists():
            return []
        images = []
        for file in sorted(OUTPUTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            # Skip keep and trash folders
            if file.is_dir() and file.name in ('keep', 'trash'):
                continue
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({
                    'name': file.name,
                    'size': file.stat().st_size,
                    'mtime': file.stat().st_mtime
                })
        return images


@app.route('/')
def index():
    """Main gallery page."""
    view = request.args.get('view', 'main')
    images = get_images(view)
    return render_template_string(GALLERY_TEMPLATE, images=images, current_view=view)


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve image files from main output directory."""
    return send_from_directory(str(OUTPUTS_DIR), filename)


@app.route('/images/keep/<path:filename>')
def serve_keep_image(filename):
    """Serve image files from keep folder."""
    return send_from_directory(str(KEEP_DIR), filename)


@app.route('/images/trash/<path:filename>')
def serve_trash_image(filename):
    """Serve image files from trash folder."""
    return send_from_directory(str(TRASH_DIR), filename)


@app.route('/api/delete', methods=['POST'])
def delete_image():
    """Move an image file to trash, or permanently delete if already in trash."""
    data = request.get_json()
    filename = data.get('filename')
    source = data.get('source', 'main')  # main, keep, or trash
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    # Determine source folder
    if source == 'keep':
        file_path = KEEP_DIR / filename
    elif source == 'trash':
        file_path = TRASH_DIR / filename
    else:
        file_path = OUTPUTS_DIR / filename
    
    # Security check: ensure file is in expected directory
    if source == 'keep':
        if not file_path.resolve().is_relative_to(KEEP_DIR.resolve()):
            return jsonify({'error': 'Invalid path'}), 400
    elif source == 'trash':
        if not file_path.resolve().is_relative_to(TRASH_DIR.resolve()):
            return jsonify({'error': 'Invalid path'}), 400
    else:
        if not file_path.resolve().is_relative_to(OUTPUTS_DIR.resolve()):
            return jsonify({'error': 'Invalid path'}), 400
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # If deleting from trash, permanently delete
        if source == 'trash':
            file_path.unlink()
            return jsonify({'success': True})
        
        # Otherwise, move to trash
        # Create trash directory if it doesn't exist
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        
        # Move file to trash folder
        dest_path = TRASH_DIR / filename
        
        # Handle filename conflicts
        counter = 1
        while dest_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            dest_path = TRASH_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.move(str(file_path), str(dest_path))
        return jsonify({'success': True, 'new_name': dest_path.name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/keep', methods=['POST'])
def keep_image():
    """Move an image to the keep folder."""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    file_path = OUTPUTS_DIR / filename
    
    # Security check: ensure file is in outputs directory
    if not file_path.resolve().is_relative_to(OUTPUTS_DIR.resolve()):
        return jsonify({'error': 'Invalid path'}), 400
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Create keep directory if it doesn't exist
        KEEP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Move file to keep folder
        dest_path = KEEP_DIR / filename
        
        # Handle filename conflicts
        counter = 1
        while dest_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            dest_path = KEEP_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.move(str(file_path), str(dest_path))
        return jsonify({'success': True, 'new_name': dest_path.name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/restore', methods=['POST'])
def restore_image():
    """Restore an image from trash back to main output."""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    file_path = TRASH_DIR / filename
    
    # Security check: ensure file is in trash directory
    if not file_path.resolve().is_relative_to(TRASH_DIR.resolve()):
        return jsonify({'error': 'Invalid path'}), 400
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Move file back to main output directory
        dest_path = OUTPUTS_DIR / filename
        
        # Handle filename conflicts
        counter = 1
        while dest_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            dest_path = OUTPUTS_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.move(str(file_path), str(dest_path))
        return jsonify({'success': True, 'new_name': dest_path.name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear-trash', methods=['POST'])
def clear_trash():
    """Permanently delete all files in trash folder."""
    if not TRASH_DIR.exists():
        return jsonify({'success': True, 'count': 0})
    
    try:
        count = 0
        for file in TRASH_DIR.iterdir():
            if file.is_file():
                file.unlink()
                count += 1
        
        return jsonify({'success': True, 'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list', methods=['GET'])
def list_images():
    """API endpoint to get current list of images."""
    view = request.args.get('view', 'main')
    images = get_images(view)
    return jsonify({'images': images})


GALLERY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ComfyUI Outputs Gallery</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
            padding: 1.5rem;
        }
        .gallery-item {
            position: relative;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s;
        }
        .gallery-item:hover {
            transform: translateY(-4px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .gallery-item img {
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }
        .gallery-item-actions {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            display: flex;
            gap: 0.5rem;
        }
        .gallery-item-actions button {
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: transform 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .gallery-item-actions button:hover {
            transform: scale(1.1);
        }
        .btn-keep {
            background: #48c774;
            color: white;
        }
        .btn-keep:hover {
            background: #3ec469;
        }
        .btn-delete {
            background: #f14668;
            color: white;
        }
        .btn-delete:hover {
            background: #e63950;
        }
        .gallery-item-info {
            padding: 0.75rem;
            font-size: 0.875rem;
            color: #4a4a4a;
        }
        .gallery-item-name {
            font-weight: 600;
            margin-bottom: 0.25rem;
            word-break: break-all;
        }
        .gallery-item-size {
            color: #7a7a7a;
            font-size: 0.75rem;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        .fade-out {
            opacity: 0;
            transition: opacity 0.3s;
        }
        .notification {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
            max-width: 400px;
        }
        .view-switcher {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            padding: 0 1.5rem;
        }
        .view-switcher button {
            flex: 1;
        }
        .btn-restore {
            background: #3273dc;
            color: white;
        }
        .btn-restore:hover {
            background: #2366d1;
        }
        .clear-trash-container {
            padding: 0 1.5rem 1rem;
        }
        .fullscreen-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 2000;
            cursor: pointer;
            align-items: center;
            justify-content: center;
        }
        .fullscreen-overlay.active {
            display: flex;
        }
        .fullscreen-image {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
            cursor: pointer;
        }
        .fullscreen-close {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: rgba(255, 255, 255, 0.2);
            border: none;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            color: white;
            font-size: 24px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
            z-index: 2001;
        }
        .fullscreen-close:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        .fullscreen-info {
            position: absolute;
            bottom: 1rem;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 0.875rem;
            z-index: 2001;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1 class="title is-3">
                <i class="fas fa-images"></i> ComfyUI Outputs Gallery
            </h1>
            <p class="subtitle is-6">
                Browse, keep, or delete your generated images from ~/ComfyUI/output
            </p>
        </div>
    </div>

    <div id="notification" class="notification is-hidden"></div>

    <!-- Fullscreen overlay -->
    <div id="fullscreenOverlay" class="fullscreen-overlay" onclick="closeFullscreen()">
        <button class="fullscreen-close" onclick="closeFullscreen(event)" title="Close (ESC)">
            <i class="fas fa-times"></i>
        </button>
        <img id="fullscreenImage" class="fullscreen-image" src="" alt="" onclick="event.stopPropagation()">
        <div id="fullscreenInfo" class="fullscreen-info"></div>
    </div>

    <div class="container">
        <div class="view-switcher">
            <button class="button {{ 'is-primary' if current_view == 'main' else 'is-light' }}" onclick="switchView('main')">
                <span class="icon"><i class="fas fa-images"></i></span>
                <span>Main</span>
            </button>
            <button class="button {{ 'is-primary' if current_view == 'keep' else 'is-light' }}" onclick="switchView('keep')">
                <span class="icon"><i class="fas fa-star"></i></span>
                <span>Keep</span>
            </button>
            <button class="button {{ 'is-primary' if current_view == 'trash' else 'is-light' }}" onclick="switchView('trash')">
                <span class="icon"><i class="fas fa-trash"></i></span>
                <span>Trash</span>
            </button>
        </div>

        {% if current_view == 'trash' %}
        <div class="clear-trash-container">
            <button class="button is-danger" onclick="clearTrash()">
                <span class="icon"><i class="fas fa-broom"></i></span>
                <span>Clear Trash</span>
            </button>
        </div>
        {% endif %}

        <div id="gallery" class="gallery">
            {% for image in images %}
            <div class="gallery-item" data-filename="{{ image.name }}">
                <div class="gallery-item-actions" onclick="event.stopPropagation()">
                    {% if current_view == 'trash' %}
                    <button class="btn-restore" onclick="restoreImage('{{ image.name }}', this)" title="Restore to main">
                        <i class="fas fa-undo"></i>
                    </button>
                    <button class="btn-delete" onclick="deleteImage('{{ image.name }}', this, 'trash')" title="Permanently delete">
                        <i class="fas fa-times"></i>
                    </button>
                    {% elif current_view == 'keep' %}
                    <button class="btn-delete" onclick="deleteImage('{{ image.name }}', this, 'keep')" title="Move to trash">
                        <i class="fas fa-times"></i>
                    </button>
                    {% else %}
                    <button class="btn-keep" onclick="keepImage('{{ image.name }}', this)" title="Move to keep folder">
                        <i class="fas fa-check"></i>
                    </button>
                    <button class="btn-delete" onclick="deleteImage('{{ image.name }}', this, 'main')" title="Move to trash">
                        <i class="fas fa-times"></i>
                    </button>
                    {% endif %}
                </div>
                <img src="/images/{{ 'keep/' if current_view == 'keep' else 'trash/' if current_view == 'trash' else '' }}{{ image.name }}" alt="{{ image.name }}" loading="lazy" onclick="openFullscreen('{{ image.name }}', '{{ 'keep/' if current_view == 'keep' else 'trash/' if current_view == 'trash' else '' }}', '{{ (image.size / 1024) | round(1) }} KB')">
                <div class="gallery-item-info">
                    <div class="gallery-item-name">{{ image.name }}</div>
                    <div class="gallery-item-size">{{ (image.size / 1024) | round(1) }} KB</div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if not images %}
        <div class="has-text-centered" style="padding: 4rem;">
            <p class="title is-4 has-text-grey">No images found</p>
            <p class="subtitle is-6 has-text-grey-light">
                {% if current_view == 'keep' %}
                No images in keep folder
                {% elif current_view == 'trash' %}
                Trash is empty
                {% else %}
                Images will appear here when you generate them in ComfyUI
                {% endif %}
            </p>
        </div>
        {% endif %}
    </div>

    <script>
        const currentView = '{{ current_view }}';

        function openFullscreen(filename, pathPrefix, size) {
            const overlay = document.getElementById('fullscreenOverlay');
            const img = document.getElementById('fullscreenImage');
            const info = document.getElementById('fullscreenInfo');
            
            img.src = `/images/${pathPrefix}${filename}`;
            info.textContent = `${filename} (${size})`;
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeFullscreen(event) {
            if (event) {
                event.stopPropagation();
            }
            const overlay = document.getElementById('fullscreenOverlay');
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        }

        // Close on ESC key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeFullscreen();
            }
        });


        function showNotification(message, type = 'success') {
            const notification = document.getElementById('notification');
            notification.className = `notification is-${type}`;
            notification.textContent = message;
            notification.classList.remove('is-hidden');
            
            setTimeout(() => {
                notification.classList.add('is-hidden');
            }, 3000);
        }

        function switchView(view) {
            window.location.href = `/?view=${view}`;
        }

        function deleteImage(filename, button, source = 'main') {
            const confirmMsg = source === 'trash' 
                ? `Permanently delete ${filename}? This cannot be undone.`
                : `Move ${filename} to trash?`;
            
            if (!confirm(confirmMsg)) {
                return;
            }
            
            const item = button.closest('.gallery-item');
            item.classList.add('fade-out');
            
            fetch('/api/delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename: filename, source: source })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    setTimeout(() => {
                        item.remove();
                        showNotification(source === 'trash' ? 'Image permanently deleted' : 'Image moved to trash', 'success');
                        checkEmptyGallery();
                    }, 300);
                } else {
                    item.classList.remove('fade-out');
                    showNotification(data.error || 'Failed to delete image', 'danger');
                }
            })
            .catch(error => {
                item.classList.remove('fade-out');
                showNotification('Error: ' + error.message, 'danger');
            });
        }

        function keepImage(filename, button) {
            const item = button.closest('.gallery-item');
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            
            fetch('/api/keep', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename: filename })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    item.classList.add('fade-out');
                    setTimeout(() => {
                        item.remove();
                        showNotification('Image moved to keep folder', 'success');
                        checkEmptyGallery();
                    }, 300);
                } else {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-check"></i>';
                    showNotification(data.error || 'Failed to move image', 'danger');
                }
            })
            .catch(error => {
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-check"></i>';
                showNotification('Error: ' + error.message, 'danger');
            });
        }

        function restoreImage(filename, button) {
            const item = button.closest('.gallery-item');
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            
            fetch('/api/restore', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename: filename })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    item.classList.add('fade-out');
                    setTimeout(() => {
                        item.remove();
                        showNotification('Image restored to main folder', 'success');
                        checkEmptyGallery();
                    }, 300);
                } else {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-undo"></i>';
                    showNotification(data.error || 'Failed to restore image', 'danger');
                }
            })
            .catch(error => {
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-undo"></i>';
                showNotification('Error: ' + error.message, 'danger');
            });
        }

        function clearTrash() {
            if (!confirm('Permanently delete all images in trash? This cannot be undone.')) {
                return;
            }
            
            fetch('/api/clear-trash', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification(`Cleared ${data.count} image(s) from trash`, 'success');
                    setTimeout(() => {
                        window.location.reload();
                    }, 500);
                } else {
                    showNotification(data.error || 'Failed to clear trash', 'danger');
                }
            })
            .catch(error => {
                showNotification('Error: ' + error.message, 'danger');
            });
        }

        function checkEmptyGallery() {
            const gallery = document.getElementById('gallery');
            if (gallery.children.length === 0) {
                const emptyMsg = currentView === 'keep' 
                    ? 'No images in keep folder'
                    : currentView === 'trash'
                    ? 'Trash is empty'
                    : 'Images will appear here when you generate them in ComfyUI';
                
                gallery.innerHTML = `
                    <div class="has-text-centered" style="grid-column: 1 / -1; padding: 4rem;">
                        <p class="title is-4 has-text-grey">No images found</p>
                        <p class="subtitle is-6 has-text-grey-light">${emptyMsg}</p>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ComfyUI Outputs Gallery")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5051, help="Port to bind (default: 5051)")
    parser.add_argument(
        "--outputs-dir",
        default=str(OUTPUTS_DIR),
        help="ComfyUI output directory (default: ~/ComfyUI/output)",
    )
    args = parser.parse_args()

    # Allow overriding output directory via CLI.
    OUTPUTS_DIR = Path(args.outputs_dir).expanduser().resolve()
    KEEP_DIR = OUTPUTS_DIR / "keep"
    TRASH_DIR = OUTPUTS_DIR / "trash"

    # Ensure outputs directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    KEEP_DIR.mkdir(parents=True, exist_ok=True)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting ComfyUI Gallery server...")
    print(f"Output directory: {OUTPUTS_DIR}")
    print(f"Keep folder: {KEEP_DIR}")
    print(f"Trash folder: {TRASH_DIR}")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run(debug=False, host=str(args.host), port=int(args.port), threaded=True)
