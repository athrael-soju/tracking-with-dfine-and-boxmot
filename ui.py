import gradio as gr
from typing import List, Any, Dict, Optional, Tuple
import pandas as pd # For type hinting the DataFrame output

import config

# Attempt to import analysis functions, handle if not available initially
try:
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
except ImportError:
    # Define placeholder functions if analysis.py is not found or has issues
    def list_dataset_files(): return ["Error: analysis.py not found or plotly missing"]
    def load_dataset(filename): return None
    def get_basic_insights(df): return {"error": "Analysis functions not available."}
    def get_track_summary_data(df): return pd.DataFrame()
    def plot_class_distribution(df): return None
    def plot_track_length_histogram(df): return None
    def plot_object_trajectories(df): return None
    def plot_unique_tracks_over_time(df): return None
    PLOTLY_AVAILABLE = False
    print("Warning: analysis.py or its dependencies (like plotly) not found or failed to import. Dataset analysis tab will not function correctly.")

# These functions will be called by create_gradio_interface
# They are moved from app.py

def create_image_inputs() -> List[gr.components.Component]:
    return [
        gr.Image(
            label="Upload Image",
            type="pil",
            sources=["upload", "webcam"],
            interactive=True,
            elem_classes="input-component",
        ),
        gr.Checkbox(label="Use Image URL Instead", value=False),
        gr.Textbox(
            label="Image URL",
            placeholder="https://example.com/image.jpg",
            visible=False,
            elem_classes="input-component",
        ),
        gr.Dropdown(
            choices=config.CHECKPOINTS,
            label="Select Model Checkpoint",
            value=config.DEFAULT_CHECKPOINT,
            elem_classes="input-component",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=config.DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.1,
            label="Confidence Threshold",
            elem_classes="input-component",
        ),
    ]


def create_video_inputs() -> List[gr.components.Component]:
    return [
        gr.Video(
            label="Upload Video",
            sources=["upload"],
            interactive=True,
            format="mp4",  # Ensure MP4 format
            elem_classes="input-component",
        ),
        gr.Dropdown(
            choices=config.CHECKPOINTS,
            label="Select Model Checkpoint",
            value=config.DEFAULT_CHECKPOINT,
            elem_classes="input-component",
        ),
        gr.Dropdown(
            choices=config.TRACKERS,
            label="Select Tracker (Optional)",
            value=None,
            elem_classes="input-component",
        ),
        gr.TextArea(
            label="Specify Class Names to Detect (comma separated)",
            value="all",
            lines=1,
            elem_classes="input-component",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=config.DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.1,
            label="Confidence Threshold",
            elem_classes="input-component",
        ),
        gr.Slider(
            minimum=1,
            maximum=30,
            value=config.DEFAULT_FPS,
            step=1,
            label="Processing FPS",
            elem_classes="input-component",
        ),
        gr.Checkbox(
            label="Create Dataset (CSV output in static/dataset)",
            value=False,
            elem_classes="input-component",
        )
    ]


def create_button_row() -> List[gr.Button]:
    return [
        gr.Button(
            f"Detect Objects", variant="primary", elem_classes="action-button"
        ),
        gr.Button(f"Clear", variant="secondary", elem_classes="action-button"),
    ]


def create_gradio_interface(process_image_fn, process_video_fn, handle_dataset_analysis_fn=None):
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        gr.Markdown(
            """
            # Object Detection Demo
            Experience state-of-the-art object detection with USTC's [D-FINE](https://huggingface.co/docs/transformers/main/model_doc/d_fine) models.
             - **Image** and **Video** modes are supported.
             - Select a model and adjust the confidence threshold to see detections!
             - On video mode, you can enable tracking powered by [Supervision](https://github.com/roboflow/supervision) and [Trackers](https://github.com/roboflow/trackers) from Roboflow.
            """,
            elem_classes="header-text",
        )

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        with gr.Group():
                            (
                                image_input,
                                use_url,
                                url_input,
                                image_model_checkpoint,
                                image_confidence_threshold,
                            ) = create_image_inputs()
                            image_detect_button, image_clear_button = create_button_row()
                    with gr.Column(scale=2):
                        image_output = gr.AnnotatedImage(
                            label="Detection Results",
                            show_label=True,
                            color_map=None,
                            elem_classes="output-component",
                        )
                gr.Examples(
                    examples=[
                        [
                            config.DEFAULT_CHECKPOINT,
                            example["path"],
                            example["url"],
                            example["use_url"],
                            config.DEFAULT_CONFIDENCE_THRESHOLD,
                        ]
                        for example in config.IMAGE_EXAMPLES
                    ],
                    inputs=[
                        image_model_checkpoint,
                        image_input,
                        url_input,
                        use_url,
                        image_confidence_threshold,
                    ],
                    outputs=[image_output],
                    fn=process_image_fn,
                    label="Select an image example to populate inputs",
                    cache_examples=True,
                    cache_mode="lazy",
                )

            with gr.Tab("Video"):
                gr.Markdown(
                    f"The input video will be processed at the selected FPS (up to {config.MAX_NUM_FRAMES} frames in result)."
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        with gr.Group():
                            video_input, video_checkpoint, video_tracker, video_classes, video_confidence_threshold, video_fps, create_dataset_checkbox = create_video_inputs()
                            video_detect_button, video_clear_button = create_button_row()
                    with gr.Column(scale=2):
                        video_output = gr.Video(
                            label="Detection Results",
                            format="mp4",  # Explicit MP4 format
                            elem_classes="output-component",
                        )

                gr.Examples(
                    examples=[
                        [example["path"], config.DEFAULT_CHECKPOINT, example["tracker"], example["classes"], config.DEFAULT_CONFIDENCE_THRESHOLD, 1, False]
                        for example in config.VIDEO_EXAMPLES
                    ],
                    inputs=[video_input, video_checkpoint, video_tracker, video_classes, video_confidence_threshold, video_fps, create_dataset_checkbox],
                    outputs=[video_output],
                    fn=process_video_fn,
                    cache_examples=False,
                    label="Select a video example to populate inputs",
                )

            with gr.Tab("Dataset Analysis"):
                gr.Markdown("Load and analyze previously generated datasets.")
                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_dropdown = gr.Dropdown(
                            label="Select Dataset CSV File",
                            choices=list_dataset_files(), # Populate dropdown
                            elem_classes="input-component"
                        )
                        analyze_button = gr.Button("Load and Analyze Dataset", variant="primary", elem_classes="action-button")
                        refresh_button = gr.Button("Refresh Dataset List", elem_classes="action-button") 
                gr.Markdown("### Numerical Insights")
                analysis_output_json = gr.JSON(label="Dataset Insights", elem_classes="output-component")
                
                gr.Markdown("### Track Summary Table")
                track_summary_table = gr.DataFrame(label="Track Details", interactive=False, wrap=True)

                if PLOTLY_AVAILABLE:
                    gr.Markdown("### Visual Insights (Interactive)")
                    with gr.Row():
                        class_dist_plot = gr.Plot(label="Class Distribution")
                        track_length_hist_plot = gr.Plot(label="Track Length Distribution (Frames)")
                    with gr.Row():
                        object_trajectories_plot = gr.Plot(label="Object Trajectories")
                        unique_tracks_plot = gr.Plot(label="Unique Active Tracks Over Time")
                else:
                    gr.Markdown("**Visual insights require Plotly to be installed for interactivity.**")

                # Define a default handler if none is passed (e.g. during standalone UI dev)
                # This default handler now needs to return multiple outputs
                def default_handle_dataset_analysis_wrapper(selected_filename: str) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
                    empty_df = pd.DataFrame()
                    if not selected_filename:
                        no_file_selected_output = {"info": "Please select a dataset file to analyze."}
                        if PLOTLY_AVAILABLE:
                            return no_file_selected_output, empty_df, None, None, None, None
                        else:
                            return no_file_selected_output, empty_df
                    
                    df = load_dataset(selected_filename)
                    if df is None or df.empty:
                        error_output = {"error": f"Failed to load or dataset is empty: {selected_filename}"}
                        if PLOTLY_AVAILABLE:
                            return error_output, empty_df, None, None, None, None
                        else:
                            return error_output, empty_df

                    insights = get_basic_insights(df)
                    summary_table_data = get_track_summary_data(df)

                    if PLOTLY_AVAILABLE:
                        fig_class_dist = plot_class_distribution(df)
                        fig_track_length_hist = plot_track_length_histogram(df)
                        fig_object_trajectories = plot_object_trajectories(df)
                        fig_unique_tracks = plot_unique_tracks_over_time(df)
                        return insights, summary_table_data, fig_class_dist, fig_track_length_hist, fig_object_trajectories, fig_unique_tracks
                    else:
                        return insights, summary_table_data
                
                actual_handle_fn = handle_dataset_analysis_fn if handle_dataset_analysis_fn else default_handle_dataset_analysis_wrapper
                
                outputs_for_click = [analysis_output_json, track_summary_table]
                if PLOTLY_AVAILABLE:
                    outputs_for_click.extend([class_dist_plot, track_length_hist_plot, object_trajectories_plot, unique_tracks_plot])

                analyze_button.click(
                    fn=actual_handle_fn,
                    inputs=[dataset_dropdown],
                    outputs=outputs_for_click
                )
                refresh_button.click(
                    fn=lambda: gr.update(choices=list_dataset_files()),
                    inputs=None,
                    outputs=[dataset_dropdown]
                )

        # Dynamic visibility for URL input
        use_url.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_url,
            outputs=url_input,
        )

        # Image clear button
        image_clear_button.click(
            fn=lambda: (
                None,
                False,
                "",
                config.DEFAULT_CHECKPOINT,
                config.DEFAULT_CONFIDENCE_THRESHOLD,
                None,
            ),
            outputs=[
                image_input,
                use_url,
                url_input,
                image_model_checkpoint,
                image_confidence_threshold,
                image_output,
            ],
        )

        # Video clear button
        video_clear_button.click(
            fn=lambda: (
                None,
                config.DEFAULT_CHECKPOINT,
                None,
                "all",
                config.DEFAULT_CONFIDENCE_THRESHOLD,
                config.DEFAULT_FPS,
                False,
            ),
            outputs=[
                video_input,
                video_checkpoint,
                video_tracker,
                video_classes,
                video_confidence_threshold,
                video_fps,
                create_dataset_checkbox,
                video_output,
            ],
        )

        # Image detect button
        image_detect_button.click(
            fn=process_image_fn,
            inputs=[
                image_model_checkpoint,
                image_input,
                url_input,
                use_url,
                image_confidence_threshold,
            ],
            outputs=[image_output],
        )

        # Video detect button
        video_detect_button.click(
            fn=process_video_fn,
            inputs=[video_input, video_checkpoint, video_tracker, video_classes, video_confidence_threshold, video_fps, create_dataset_checkbox],
            outputs=[video_output],
        )
    return demo 