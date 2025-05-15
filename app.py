import logging
import torch
from typing import Dict, Any, Optional, Tuple
import pandas as pd # For type hinting and empty DataFrame

from pathlib import Path

import config
from ui import create_gradio_interface
from processing import process_image, process_video
from analysis import (
    list_dataset_files, 
    load_dataset, 
    get_basic_insights,
    get_track_summary_data,
    plot_class_distribution,
    plot_track_length_histogram,
    plot_object_trajectories,
    plot_unique_tracks_over_time,
    PLOTLY_AVAILABLE
)

# Configuration constants
TORCH_DTYPE = torch.float32

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def handle_dataset_analysis(selected_filename: str):
    """
    Handles loading and analyzing dataset files from static/dataset/
    
    Args:
        selected_filename (str): Name of the CSV file to analyze
        
    Returns:
        If PLOTLY_AVAILABLE:
            (insights_dict, track_summary_df, class_dist_plot, track_length_hist, object_trajectories_plot, unique_tracks_plot)
        Else:
            (insights_dict, track_summary_df)
    """
    empty_df = pd.DataFrame() # For returning in case of errors or no selection
    if not selected_filename:
        output_dict = {"info": "Please select a dataset file to analyze."}
        if PLOTLY_AVAILABLE:
            return output_dict, empty_df, None, None, None, None
        else:
            return output_dict, empty_df

    try:
        df = load_dataset(selected_filename)
        if df is None or df.empty:
            output_dict = {"error": f"Failed to load or dataset is empty: {selected_filename}"}
            if PLOTLY_AVAILABLE:
                return output_dict, empty_df, None, None, None, None
            else:
                return output_dict, empty_df

        insights = get_basic_insights(df)
        track_summary_df = get_track_summary_data(df)
        
        if PLOTLY_AVAILABLE:
            fig_class_dist = plot_class_distribution(df)
            fig_track_length_hist = plot_track_length_histogram(df)
            fig_object_trajectories = plot_object_trajectories(df)
            fig_unique_tracks = plot_unique_tracks_over_time(df)
            return insights, track_summary_df, fig_class_dist, fig_track_length_hist, fig_object_trajectories, fig_unique_tracks
        else:
            return insights, track_summary_df

    except FileNotFoundError:
        output_dict = {"error": f"Dataset file not found: {selected_filename}"}
        if PLOTLY_AVAILABLE:
            return output_dict, empty_df, None, None, None, None
        else:
            return output_dict, empty_df
    except Exception as e:
        output_dict = {"error": f"An error occurred during analysis: {str(e)}"}
        if PLOTLY_AVAILABLE:
            return output_dict, empty_df, None, None, None, None
        else:
            return output_dict, empty_df

if __name__ == "__main__":
    demo = create_gradio_interface(
        process_image_fn=process_image, 
        process_video_fn=process_video,
        handle_dataset_analysis_fn=handle_dataset_analysis
    )
    demo.queue(max_size=20).launch()
