import torch
import tqdm
import spaces
import numpy as np
import gradio as gr
from functools import lru_cache
from typing import List, Optional, Tuple

from transformers import AutoModelForObjectDetection, AutoImageProcessor

import config

# Configuration constants
TORCH_DTYPE = torch.float32 # Assuming this is used by the model loading

@lru_cache(maxsize=3)
def get_model_and_processor(checkpoint: str):
    model = AutoModelForObjectDetection.from_pretrained(checkpoint, torch_dtype=TORCH_DTYPE)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return model, image_processor


@spaces.GPU(duration=20) # Assuming spaces.GPU is relevant here
def detect_objects(
    checkpoint: str,
    images: List[np.ndarray] | np.ndarray,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    target_size: Optional[Tuple[int, int]] = None,
    batch_size: int = config.BATCH_SIZE,
    classes: Optional[List[str]] = None,
):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, image_processor = get_model_and_processor(checkpoint)
    model = model.to(device)

    if classes is not None:
        wrong_classes = [cls for cls in classes if cls not in model.config.label2id]
        if wrong_classes:
            # Consider if gr.Warning should be handled differently in a utility module
            # For now, keeping it, but it might be better to raise an exception or return info
            gr.Warning(f"Classes not found in model config: {wrong_classes}")
        keep_ids = [model.config.label2id[cls] for cls in classes if cls in model.config.label2id]
    else:
        keep_ids = None

    if isinstance(images, np.ndarray) and images.ndim == 4:
        images = [x for x in images]  # split video array into list of images

    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

    results = []
    for batch in tqdm.tqdm(batches, desc="Processing frames"):

        # preprocess images
        inputs = image_processor(images=batch, return_tensors="pt")
        inputs = inputs.to(device).to(TORCH_DTYPE)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # postprocess outputs
        if target_size:
            target_sizes = [target_size] * len(batch)
        else:
            target_sizes = [(image.shape[0], image.shape[1]) for image in batch]

        batch_results = image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )

        results.extend(batch_results)

    # move results to cpu
    for i, result in enumerate(results):
        results[i] = {k: v.cpu() for k, v in result.items()}
        if keep_ids is not None:
            keep = torch.isin(results[i]["labels"], torch.tensor(keep_ids))
            results[i] = {k: v[keep] for k, v in results[i].items()}
    
    return results, model.config.id2label 