import os
import argparse
import cv2
import torch
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import SamPredictor, sam_model_registry

def main():
    parser = argparse.ArgumentParser(description="Detect and segment windows in an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--text_prompt", type=str, default="window", help="Text prompt for GroundingDINO.")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box threshold for GroundingDINO.")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold for GroundingDINO.")
    parser.add_argument("--dino_weights", type=str, default="groundingdino_swinb.pth", help="Path to GroundingDINO weights.")
    parser.add_argument("--dino_config", type=str, default="GroundingDINO_SwinB_cfg.py", help="Path to GroundingDINO config.")
    parser.add_argument("--sam_weights", type=str, default="sam_vit_b.pth", help="Path to SAM weights.")
    parser.add_argument("--sam_type", type=str, default="vit_b", help="SAM model type.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Force device selection (cpu/cuda). Defaults to auto-detect.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return

    if not os.path.exists(args.dino_weights):
        print(f"Error: GroundingDINO weights '{args.dino_weights}' not found. Please download them.")
        return

    if not os.path.exists(args.dino_config):
        print(f"Error: GroundingDINO config '{args.dino_config}' not found.")
        return
    
    if not os.path.exists(args.sam_weights):
        print(f"Error: SAM weights '{args.sam_weights}' not found. Please download them.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    try:
        # GroundingDINO load_model returns the model directly
        dino_model = load_model(model_config_path=args.dino_config, model_checkpoint_path=args.dino_weights)
    except Exception as e:
        print(f"Error loading GroundingDINO model: {e}")
        print("Ensure you have the correct dependencies and model weights.")
        return

    # Determine device robustly
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Additional CUDA capability check: try a tiny tensor op
    if device == "cuda":
        try:
            _ = torch.tensor([1.0], device="cuda") * 2
        except Exception as e:
            print(f"CUDA runtime check failed: {e}. Falling back to CPU.")
            device = "cpu"

    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_weights)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    print(f"Processing {args.image}...")
    # GroundingDINO load_image returns (image_source, image_transformed)
    # image_source is numpy array (H, W, 3), image_transformed is Tensor
    image_source, image_transformed = load_image(args.image)

    # Step 1: Detect windows
    # predict returns (boxes, logits, phrases)
    # boxes are normalized (cx, cy, w, h)
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_transformed,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    print(f"Found {len(boxes)} potential windows.")

    # Step 2: Segment with SAM
    # SAM expects RGB image (numpy)
    sam_predictor.set_image(image_source)
    
    # Convert normalized boxes (cx, cy, w, h) to (x1, y1, x2, y2) for SAM
    h, w, _ = image_source.shape
    boxes_xyxy = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = torch.cat([
        boxes_xyxy[:, :2] - boxes_xyxy[:, 2:] / 2,
        boxes_xyxy[:, :2] + boxes_xyxy[:, 2:] / 2
    ], dim=1)

    for i, box in enumerate(boxes_xyxy):
        # SAM predict expects box as [x1, y1, x2, y2]
        box_np = box.numpy()
        
        masks, scores, logits = sam_predictor.predict(box=box_np, multimask_output=False)
        
        # Save cropped window
        x1, y1, x2, y2 = map(int, box_np)
        
        # Clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue

        # Use image_source (RGB) but convert to BGR for OpenCV saving
        crop = image_source[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        
        output_filename = os.path.join(args.output_dir, f"window_crop_{i}.png")
        cv2.imwrite(output_filename, crop_bgr)
        print(f"Saved {output_filename}")

if __name__ == "__main__":
    main()
