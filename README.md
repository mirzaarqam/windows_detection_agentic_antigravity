# Window Detection with GroundingDINO and SAM

This project detects and segments windows in images using GroundingDINO for detection and Segment Anything Model (SAM) for segmentation.

## Setup

1.  **Install Basic Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Model Libraries**:
    It is recommended to install these separately to ensure `torch` is available during their build process.
    ```bash
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```
    *Note: If you encounter errors, ensure you have a CUDA-capable GPU and the correct CUDA toolkit installed for `GroundingDINO`.*

2.  **Download Model Weights**:
    You need to download the following model weights and place them in the project root:
    *   `groundingdino_swinb.pth`: [Download from GroundingDINO releases](https://github.com/IDEA-Research/GroundingDINO/releases)
    *   `sam_vit_b.pth`: [Download from SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
    *   `GroundingDINO_SwinB_cfg.py`: Configuration file for GroundingDINO (usually found in the GroundingDINO repo).

## Usage

Run the detection script:

```bash
python detect_window.py --image path/to/your/image.jpg
```

### Arguments
*   `--image`: Path to the input image (required).
*   `--text_prompt`: Text prompt for detection (default: "window").
*   `--box_threshold`: Threshold for bounding box confidence (default: 0.35).
*   `--text_threshold`: Threshold for text confidence (default: 0.25).
*   `--output_dir`: Directory to save results (default: "output").

## Troubleshooting

### `ImportError: cannot import name 'Model' from 'groundingdino.util.inference'`
The provided code uses a wrapper class `Model` that might not be present in the official `groundingdino` repository. If you encounter this error, you may need to adjust the import in `detect_window.py` to use `load_model`, `load_image`, and `predict` functions directly from `groundingdino.util.inference`, or ensure you are using the specific fork/wrapper intended by the code source.

## Output
The script will save cropped images of detected windows in the `output` directory.
