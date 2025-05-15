import gradio as gr
from typing import List

import config

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
    ]


def create_button_row() -> List[gr.Button]:
    return [
        gr.Button(
            f"Detect Objects", variant="primary", elem_classes="action-button"
        ),
        gr.Button(f"Clear", variant="secondary", elem_classes="action-button"),
    ]


def create_gradio_interface(process_image_fn, process_video_fn):
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
                            video_input, video_checkpoint, video_tracker, video_classes, video_confidence_threshold, video_fps = create_video_inputs()
                            video_detect_button, video_clear_button = create_button_row()
                    with gr.Column(scale=2):
                        video_output = gr.Video(
                            label="Detection Results",
                            format="mp4",  # Explicit MP4 format
                            elem_classes="output-component",
                        )

                gr.Examples(
                    examples=[
                        [example["path"], config.DEFAULT_CHECKPOINT, example["tracker"], example["classes"], config.DEFAULT_CONFIDENCE_THRESHOLD, 1] # Added default FPS
                        for example in config.VIDEO_EXAMPLES
                    ],
                    inputs=[video_input, video_checkpoint, video_tracker, video_classes, video_confidence_threshold, video_fps],
                    outputs=[video_output],
                    fn=process_video_fn,
                    cache_examples=False,
                    label="Select a video example to populate inputs",
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
                None,
            ),
            outputs=[
                video_input,
                video_checkpoint,
                video_tracker,
                video_classes,
                video_confidence_threshold,
                video_fps, # Added video_fps to outputs
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
            inputs=[video_input, video_checkpoint, video_tracker, video_classes, video_confidence_threshold, video_fps],
            outputs=[video_output],
        )
    return demo 