import torch
import numpy as np
import supervision as sv
from boxmot import BotSort
from pathlib import Path

import trackers # This might be a local module, ensure it's available or adjust path
import config # Assuming config.py contains TrackingAlgorithm

# It's good practice to define constants or enums in a central place like config.py
# For example, config.TrackingAlgorithm.SORT, config.TrackingAlgorithm.DEEPSORT, etc.

def get_tracker(tracker: str, fps: float):
    if tracker == config.TrackingAlgorithm.SORT:
        return trackers.SORTTracker(frame_rate=fps)
    elif tracker == config.TrackingAlgorithm.DEEPSORT:
        # Ensure DeepSORTFeatureExtractor is correctly imported or defined
        # If it's part of the 'trackers' module, it might be trackers.DeepSORTFeatureExtractor
        feature_extractor = trackers.DeepSORTFeatureExtractor.from_timm("mobilenetv4_conv_small.e1200_r224_in1k", device="cpu")
        return trackers.DeepSORTTracker(feature_extractor, frame_rate=fps)
    elif tracker == config.TrackingAlgorithm.BYTETRACK:
        return sv.ByteTrack(frame_rate=int(fps))
    elif tracker == config.TrackingAlgorithm.BOTSORT_OSNET:
        return BotSort(
            reid_weights=Path("osnet_x1_0_msmt17.pt"), # Ensure this path is correct or configurable
            device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
            half=False,
            per_class=False
        )
    elif tracker == config.TrackingAlgorithm.BOTSORT_CLIP:
        return BotSort(
            reid_weights=Path("clip_market1501.pt"), # Ensure this path is correct or configurable
            device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
            half=False,
            per_class=False
        )
    else:
        raise ValueError(f"Invalid tracker: {tracker}")


def update_tracker(tracker, detections: sv.Detections, frame: np.ndarray):
    tracker_name = tracker.__class__.__name__
    if tracker_name == "SORTTracker":
        return tracker.update(detections)
    elif tracker_name == "DeepSORTTracker":
        return tracker.update(detections, frame)
    elif tracker_name == "ByteTrack":
        return tracker.update_with_detections(detections)
    elif tracker_name == "BotSort":
        if detections.xyxy.shape[0] > 0 :
            valid_indices = [i for i, cid in enumerate(detections.class_id) if cid is not None]
            
            if not valid_indices:
                # If no valid class_ids (e.g., all are None), create an empty array for BotSort
                dets_np = np.empty((0, 6))
            else:
                # Filter detections to include only those with non-None class_id
                xyxy = detections.xyxy[valid_indices]
                confidence = detections.confidence[valid_indices]
                class_id = detections.class_id[valid_indices].astype(int) # Ensure class_id is int
                
                # BotSort expects an array of shape (n, 6) with [x1, y1, x2, y2, conf, cls_id]
                dets_np = np.column_stack((
                    xyxy,
                    confidence,
                    class_id
                ))
        else:
            # No detections, pass an empty array to BotSort
            dets_np = np.empty((0, 6))

        tracked_objects = tracker.update(dets_np, frame) # Pass the original frame
        
        # Handle case where BotSort returns no tracked objects
        if tracked_objects.shape[0] == 0:
            return sv.Detections.empty()

        # Convert BotSort output back to sv.Detections
        return sv.Detections(
            xyxy=tracked_objects[:, :4],
            # confidence=tracked_objects[:, 4], # Original BotSort output might have track score here
            confidence=tracked_objects[:, 5], # Assuming confidence is the 6th column (index 5)
            class_id=tracked_objects[:, 6].astype(int), # Assuming class_id is the 7th column (index 6)
            tracker_id=tracked_objects[:, 4].astype(int) # Assuming tracker_id is the 5th column (index 4)
        )
    else:
        raise ValueError(f"Invalid tracker: {tracker_name}") 