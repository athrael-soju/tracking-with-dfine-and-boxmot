import os
import cv2
import tqdm
import numpy as np
import gradio as gr
import imageio.v3 as iio
import supervision as sv
from PIL import Image
from pathlib import Path # For VIDEO_OUTPUT_DIR
from typing import Optional, Tuple, List # For type hints

from transformers.image_utils import load_image

import config
# Assuming these are from other refactored modules, adjust paths if necessary
from object_detection import detect_objects
from tracking import get_tracker, update_tracker

# It might be good to move VIDEO_OUTPUT_DIR and color to config.py or pass them as arguments
# For now, keeping them here for simplicity during refactoring
VIDEO_OUTPUT_DIR = Path(config.VIDEO_OUTPUT_DIR_STR)
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
color = sv.ColorPalette.from_hex(config.COLOR_PALETTE_HEX)

# Logger might also be passed or initialized per module
import logging
logger = logging.getLogger(__name__)

def process_image(
    checkpoint: str = config.DEFAULT_CHECKPOINT,
    image: Optional[Image.Image] = None,
    url: Optional[str] = None,
    use_url: bool = False,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    # logger instance to be passed, or use a global one if defined appropriately
):
    if not use_url:
        url = None

    if (image is None) ^ bool(url):
        logger.warning("Exclusive OR condition for image and url not met. Proceeding, but this may cause issues.")
        if image is None and url is None:
             raise gr.Error("Please upload an image or provide an image URL.")

    if url:
        try:
            image = load_image(url)
        except Exception as e:
            raise gr.Error(f"Failed to load image from URL: {url}. Error: {e}")
    elif image is None:
        raise gr.Error("No image provided.")

    results, id2label = detect_objects(
        checkpoint=checkpoint,
        images=[np.array(image)],
        confidence_threshold=confidence_threshold,
    )
    result = results[0]

    annotations = []
    for label, score, box in zip(result["labels"], result["scores"], result["boxes"]):
        text_label = id2label[label.item()]
        formatted_label = f"{text_label} ({score:.2f})"
        x_min, y_min, x_max, y_max = box.cpu().numpy().round().astype(int)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.width - 1, x_max)
        y_max = min(image.height - 1, y_max)
        annotations.append(((x_min, y_min, x_max, y_max), formatted_label))

    return (image, annotations)

def get_target_size(image_height, image_width, max_size: int):
    if image_height < max_size and image_width < max_size:
        new_height, new_width = image_height, image_width 
    elif image_height > image_width:
        new_height = max_size
        new_width = int(image_width * max_size / image_height)
    else:
        new_width = max_size
        new_height = int(image_height * max_size / image_width)
    
    new_height = new_height // 2 * 2
    new_width = new_width // 2 * 2

    return new_width, new_height

def read_video_k_frames(video_path: str, k: int, read_every_i_frame: int = 1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    progress_bar = tqdm.tqdm(total=k, desc="Reading frames")
    while cap.isOpened() and len(frames) < k:
        ret, frame = cap.read()
        if not ret:
            break
        if i % read_every_i_frame == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            progress_bar.update(1)
        i += 1
    cap.release()
    progress_bar.close()
    return frames

def process_video(
    video_path: str,
    checkpoint: str,
    tracker_algorithm: Optional[str] = None,
    classes: str = "all",
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    video_fps: int = 1,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
    # VIDEO_OUTPUT_DIR and color can be passed as args if they are not global or part of config
) -> str:
    if not video_path or not os.path.isfile(video_path):
        raise gr.Error(f"Invalid video path: {video_path}. Please upload a valid video file.")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        raise gr.Error(f"Unsupported video format: {ext}, supported formats: {config.ALLOWED_VIDEO_EXTENSIONS}")

    video_info = sv.VideoInfo.from_video_path(video_path)

    # video_fps is the user's desired output FPS (int, >0 from Gradio slider)
    # video_info.fps is source FPS (float, assumed >0 by sv.VideoInfo)
    if video_fps <= 0:
        # This case should ideally not be reached due to Gradio slider constraints (min=1)
        raise gr.Error("Processing FPS must be positive.")
    if video_info.fps <= 0:
        # If source FPS is unknown or zero, this indicates an issue with the video or video_info.
        # Fallback: attempt to process frame by frame, and use user's desired FPS for output.
        # Tracker might not behave optimally.
        logger.warning(f"Source video FPS is {video_info.fps}. Proceeding with caution.")
        read_each_i_frame = 1
        actual_processing_fps = float(video_fps) # Tracker gets the target output FPS
    else:
        read_each_i_frame = max(1, int(round(video_info.fps / video_fps)))
        # Effective FPS of the frames being processed (for the tracker)
        actual_processing_fps = video_info.fps / read_each_i_frame

    target_width, target_height = get_target_size(video_info.height, video_info.width, 1080)

    n_frames_to_read = video_info.total_frames // read_each_i_frame
    frames = read_video_k_frames(video_path, n_frames_to_read, read_each_i_frame)
    frames = [cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC) for frame in frames]

    color_lookup = sv.ColorLookup.TRACK if tracker_algorithm else sv.ColorLookup.CLASS
    box_annotator = sv.BoxAnnotator(color, color_lookup=color_lookup, thickness=1)
    label_annotator = sv.LabelAnnotator(color, color_lookup=color_lookup, text_scale=0.5)
    trace_annotator = sv.TraceAnnotator(color, color_lookup=color_lookup, thickness=1, trace_length=100)

    if classes != "all":
        classes_list = [cls.strip() for cls in classes.split(",")]
    else:
        classes_list = None

    results, id2label = detect_objects(
        images=np.array(frames),
        checkpoint=checkpoint,
        confidence_threshold=confidence_threshold,
        target_size=(target_height, target_width),
        classes=classes_list,
    )

    annotated_frames = []
    if tracker_algorithm:
        tracker = get_tracker(tracker_algorithm, actual_processing_fps) # Use actual_processing_fps for tracker
        for frame, result in progress.tqdm(zip(frames, results), desc="Tracking objects", total=len(frames)):
            detections = sv.Detections.from_transformers(result, id2label=id2label)
            detections = detections.with_nms(threshold=0.95, class_agnostic=True)
            detections = update_tracker(tracker, detections, frame)
            labels = []
            if detections.tracker_id is not None:
                labels = [
                    f"#{tracker_id} {id2label[class_id]} ({confidence:.2f})"
                    for class_id, tracker_id, confidence in zip(detections.class_id, detections.tracker_id, detections.confidence)
                ]
            elif detections.class_id is not None and detections.confidence is not None: # Ensure other necessary fields are present
                labels = [
                    f"{id2label[class_id]} ({confidence:.2f})"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]
            # If even class_id or confidence is None, labels will remain empty, which is a safe fallback.

            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            if detections.tracker_id is not None:
                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frames.append(annotated_frame)
    else:
        for frame, result in tqdm.tqdm(zip(frames, results), desc="Annotating frames", total=len(frames)):
            detections = sv.Detections.from_transformers(result, id2label=id2label)
            detections = detections.with_nms(threshold=0.95, class_agnostic=True)
            labels = [
                f"{id2label[class_id]} ({confidence:.2f})"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            annotated_frames.append(annotated_frame)

    base, ext = os.path.splitext(os.path.basename(video_path))
    output_filename = os.path.join(VIDEO_OUTPUT_DIR, f"{base}_processed.mp4")
    iio.imwrite(output_filename, annotated_frames, fps=video_fps, codec="h264") # Use user-selected video_fps for output file
    return output_filename 