# Object Detection and Tracking Web Application

This project provides a user-friendly web interface for performing object detection on images and object detection combined with tracking on videos. It leverages state-of-the-art models for detection and multiple popular algorithms for tracking.

## Features

-   **Web Interface:** Easy-to-use interface built with Gradio for uploading images/videos and configuring parameters.
-   **Object Detection:**
    -   Utilizes "dfine" models (e.g., `dfine-medium-obj365`, `dfine-large-coco`) from Hugging Face for accurate object detection.
    -   Process local images or images from URLs.
    -   Adjustable confidence threshold.
    -   Option to filter detections by specific class names.
-   **Object Tracking:**
    -   Supports multiple tracking algorithms:
        -   ByteTrack
        -   DeepSORT
        -   SORT
        -   BoTSORT (with OSNet or CLIP Re-ID models)
    -   Processes uploaded videos.
    -   Visualizes tracks with bounding boxes, labels, and trace lines.
    -   Adjustable output video FPS.
-   **Customization:**
    -   Select different object detection model checkpoints.
    -   Choose your preferred tracking algorithm for video processing.

## Project Structure

Key files and directories:

-   `app.py`: Main application script that launches the Gradio interface.
-   `ui.py`: Defines the Gradio user interface components and layout.
-   `processing.py`: Contains the core logic for image and video processing, including detection and tracking pipelines.
-   `object_detection.py`: Handles loading detection models (from Hugging Face `transformers`) and performing inference.
-   `tracking.py`: Implements the integration of various tracking algorithms (`supervision`, `boxmot`).
-   `config.py`: Stores configuration constants, such as model checkpoints, default parameters, and example files.
-   `requirements.txt`: Lists the Python dependencies.
-   `examples/`: Contains sample images and videos for testing.
-   `static/`: Directory where processed output videos are saved.

## Getting Started

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)
-   For GPU acceleration (recommended for video processing), an NVIDIA GPU with CUDA and cuDNN installed, or an Apple Silicon Mac with Metal support.

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```

3.  **Install dependencies:**
    The project uses PyTorch. Please install it first according to your system and CUDA version (if applicable) from the [official PyTorch website](https://pytorch.org/get-started/locally/).

    Then, install the remaining requirements:
    ```bash
    pip install -r requirements.txt
    ```

    **Note for Apple Silicon (M1/M2 Macs):**
    If you encounter issues installing `numpy` or other packages, you might need to set specific environment variables or install dependencies like OpenBLAS manually. The original template mentioned:
    ```bash
    brew install openblas
    OPENBLAS="$(brew --prefix openblas)" pip install numpy
    ```
    Ensure PyTorch is installed correctly for MPS (Metal Performance Shaders) support. Refer to PyTorch documentation for Apple Silicon.

4.  **Download Re-ID model weights (for BoTSORT and DeepSORT):**
    The `BoTSORT` and `DeepSORT` trackers require pre-trained Re-ID model weights.
    -   For `BoTSORT` with OSNet: Download `osnet_x1_0_msmt17.pt` and place it in the project's root directory or update the path in `tracking.py`.
    -   For `BoTSORT` with CLIP: Download `clip_market1501.pt` and place it in the project's root directory or update the path in `tracking.py`.
    -   DeepSORT also requires a Re-ID model (e.g., `mobilenetv4_conv_small.e1200_r224_in1k` used in `tracking.py`). Ensure this model is accessible or download the weights if necessary. `trackers.DeepSORTFeatureExtractor.from_timm` should handle downloading if the model is available via `timm`.

    *(You may need to find the sources for these weight files if they are not automatically downloaded by the libraries.)*

### Running the Application

Once the installation is complete, run the application:

```bash
python app.py
```

This will start the Gradio web server, and you can access the interface by navigating to the URL printed in your console (usually `http://127.0.0.1:7860`).

## Usage

1.  **Open the web interface** in your browser.
2.  **For Image Processing:**
    -   Go to the "Image Detection" tab.
    -   Upload an image file or provide a URL.
    -   Select the desired detection model checkpoint.
    -   Adjust the confidence threshold if needed.
    -   Click "Detect Objects". The processed image with detections will be displayed.
3.  **For Video Processing:**
    -   Go to the "Video Tracking" tab.
    -   Upload a video file.
    -   Select the desired detection model checkpoint.
    -   Choose a tracking algorithm from the dropdown (or select "None" for detection only).
    -   Optionally, specify a comma-separated list of class names to focus on (e.g., "person, car, dog"). Use "all" to detect all classes.
    -   Adjust the confidence threshold and output video FPS.
    -   Click "Process Video". The processing might take some time depending on the video length and hardware.
    -   Once finished, the processed video will be available for viewing and download.

## Dependencies

Key libraries used:

-   [Gradio](https.gradio.app/): For creating the web UI.
-   [PyTorch](https://pytorch.org/): For deep learning models.
-   [Hugging Face Transformers](https://huggingface.co/docs/transformers/index): For loading pre-trained object detection models.
-   [Supervision](https://roboflow.github.io/supervision/): For computer vision utilities, including annotations, NMS, and ByteTrack.
-   [BoxMOT](https://github.com/FengText/BoxMOT): For BoTSORT, DeepSORT, and SORT tracking algorithms.
-   OpenCV (`cv2`): For image and video manipulation.
-   NumPy: For numerical operations.
-   ImageIO: For writing video files.

See `requirements.txt` for a full list of dependencies.

## Future Improvements / To-Do

-   [ ] Add download links for required Re-ID model weights or integrate automatic download.
-   [ ] More robust error handling and user feedback.
-   [ ] Option for selecting specific Re-ID models for DeepSORT/BoTSORT through the UI.
-   [ ] Performance optimization for video processing (e.g., more efficient frame handling).
-   [ ] Support for more detection models and tracking algorithms.
-   [ ] Unit and integration tests.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

[Guidelines for contributing to the project - You can expand this section if needed]

## License

[Specify the project license, e.g., MIT License]

*This README was generated with assistance from an AI coding assistant.*