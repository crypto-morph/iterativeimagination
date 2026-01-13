"""Constants used throughout the Iterative Imagination codebase."""

# Mask analysis thresholds
MASK_WHITE_THRESHOLD = 0.95  # Percentage of white pixels to consider mask "all white"
MASK_WHITE_PIXEL_VALUE = 240  # Grayscale value threshold for white pixels
MASK_COVERAGE_WARNING_THRESHOLD = 50.0  # Percentage coverage to warn about

# Parameter defaults
DEFAULT_DENOISE = 0.5
DEFAULT_CFG = 7.0
DEFAULT_STEPS = 25
DEFAULT_SAMPLER = "dpmpp_2m"
DEFAULT_SCHEDULER = "karras"

# Parameter bounds
DENOISE_MIN = 0.20
DENOISE_MAX = 0.80
DENOISE_MAX_NO_MASK = 0.55
CFG_MIN = 4.0
CFG_MAX = 12.0
CFG_MAX_NO_MASK = 9.0

# Inpainting boost defaults
INPAINTING_DENOISE_MIN = 0.85
INPAINTING_CFG_MIN = 10.0
INPAINTING_DENOISE_THRESHOLD = 0.7
INPAINTING_CFG_THRESHOLD = 9.0

# AI parameter recommendation confidence
CONFIDENCE_THRESHOLD = 0.5

# Similarity thresholds
SIMILARITY_TOO_LOW_THRESHOLD = 0.30
SIMILARITY_REDUCTION_DELTA = 0.05

# Parameter adjustment deltas
DENOISE_DELTA_BASE = 0.06
CFG_DELTA_BASE = 0.5
DENOISE_DELTA_PRESERVE = 0.05
CFG_DELTA_PRESERVE = 0.5

# High parameter warning thresholds
HIGH_DENOISE_THRESHOLD = 0.85
HIGH_CFG_THRESHOLD = 12.0

# Image processing
IMAGE_MAX_SIZE = 1024  # Max dimension for base64 encoding

# ComfyUI
COMFYUI_DEFAULT_HOST = "localhost"
COMFYUI_DEFAULT_PORT = 8188
COMFYUI_TIMEOUT = 30
COMFYUI_DOWNLOAD_TIMEOUT = 60
COMFYUI_WAIT_TIMEOUT = 300

# AIVis
AIVIS_DEFAULT_TIMEOUT = 180
AIVIS_RETRY_ATTEMPTS = 2
AIVIS_RETRY_DELAY = 5
