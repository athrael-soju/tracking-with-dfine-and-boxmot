import logging
import torch

from pathlib import Path

import config
from ui import create_gradio_interface
from processing import process_image, process_video

# Configuration constants
TORCH_DTYPE = torch.float32

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    demo = create_gradio_interface(process_image_fn=process_image, process_video_fn=process_video)
    demo.queue(max_size=20).launch()
