from pathlib import Path

# Configuration constants
CHECKPOINTS = [
    "ustc-community/dfine-medium-obj2coco",
    "ustc-community/dfine-medium-coco",
    "ustc-community/dfine-medium-obj365",
    "ustc-community/dfine-nano-coco",
    "ustc-community/dfine-small-coco",
    "ustc-community/dfine-large-coco",
    "ustc-community/dfine-xlarge-coco",
    "ustc-community/dfine-small-obj365",
    "ustc-community/dfine-large-obj365",
    "ustc-community/dfine-xlarge-obj365",
    "ustc-community/dfine-small-obj2coco",
    "ustc-community/dfine-large-obj2coco-e25",
    "ustc-community/dfine-xlarge-obj2coco",
]
DEFAULT_CHECKPOINT = CHECKPOINTS[9]
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

# Image
IMAGE_EXAMPLES = [
    {"path": "./examples/images/tennis.jpg", "use_url": False, "url": "", "label": "Local Image"},
    {"path": "./examples/images/dogs.jpg", "use_url": False, "url": "", "label": "Local Image"},
    {"path": "./examples/images/nascar.jpg", "use_url": False, "url": "", "label": "Local Image"},
    {"path": "./examples/images/crossroad.jpg", "use_url": False, "url": "", "label": "Local Image"},
    {
        "path": None,
        "use_url": True,
        "url": "https://live.staticflickr.com/65535/33021460783_1646d43c54_b.jpg",
        "label": "Flickr Image",
    },
]

# Video
MAX_NUM_FRAMES = 250
BATCH_SIZE = 4
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
VIDEO_OUTPUT_DIR_STR = "static/videos"
DEFAULT_FPS = 1


class TrackingAlgorithm:
    BYTETRACK = "ByteTrack (2021)"
    DEEPSORT = "DeepSORT (2017)"
    SORT = "SORT (2016)"
    BOTSORT_OSNET = "BoTSORT (OSNet x1.0 MSMT17)"
    BOTSORT_CLIP = "BoTSORT (CLIP Market1501)"

TRACKERS = [None, TrackingAlgorithm.BYTETRACK, TrackingAlgorithm.DEEPSORT, TrackingAlgorithm.SORT, TrackingAlgorithm.BOTSORT_OSNET, TrackingAlgorithm.BOTSORT_CLIP]

VIDEO_EXAMPLES = [
    {"path": "./examples/videos/dogs_running.mp4", "label": "Local Video", "tracker": None, "classes": "all"},
    {"path": "./examples/videos/traffic.mp4", "label": "Local Video", "tracker": TrackingAlgorithm.BOTSORT_CLIP, "classes": "Car, Truck, Van"},
    {"path": "./examples/videos/fast_and_furious.mp4", "label": "Local Video", "tracker": None, "classes": "all"},
    {"path": "./examples/videos/break_dance.mp4", "label": "Local Video", "tracker": None, "classes": "all"},
]

# Color palette for visualization
COLOR_PALETTE_HEX = [
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
] 