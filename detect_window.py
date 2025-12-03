import os
import argparse
import cv2
import torch
from groundingdino.util.inference import Model as GroundingDINO
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
    # Load models
    try:
        dino = GroundingDINO(model_config_path=args.dino_config, model_checkpoint_path=args.dino_weights)
    except Exception as e:
        print(f"Error loading GroundingDINO model: {e}")
        print("Ensure you have the correct dependencies and model weights.")
        return

    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_weights)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(sam)

    print(f"Processing {args.image}...")
    img = cv2.imread(args.image)
    if img is None:
        print("Failed to load image.")
        return

    # Step 1: Detect windows
    # predict method signature in user code: predict(img, text_prompt="window")
    # Standard GroundingDINO predict usually takes (image, caption, box_threshold, text_threshold, device)
    # The wrapper `Model` likely handles this.
    boxes = dino.predict_with_caption(image=img, caption=args.text_prompt, box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    # Note: I changed `predict` to `predict_with_caption` or similar if I was writing from scratch, 
    # but user used `dino.predict`. I will stick to `dino.predict` if that's what their wrapper does.
    # However, to be helpful, I will assume the user might be using a specific library like `groundingdino-py` or similar.
    # Let's stick to the user's method name `predict` but pass the extra args if accepted, or just the prompt.
    # User code: `boxes = dino.predict(img, text_prompt="window")`
    
    # I'll use the user's exact call for `dino.predict` to avoid breaking their specific wrapper, 
    # but I'll add the thresholds if the method supports **kwargs or if I can inspect it. 
    # Since I can't inspect, I'll trust the user's snippet for the method name but add the thresholds to the call if possible.
    # Actually, `dino.predict(img, text_prompt="window")` is very specific. 
    # I will use that.
    
    boxes = dino.predict(img, text_prompt=args.text_prompt)

    print(f"Found {len(boxes)} potential windows.")

    # Step 2: Segment with SAM
    sam_predictor.set_image(img)
    
    for i, box in enumerate(boxes):
        # SAM predict expects box as [x1, y1, x2, y2]
        # GroundingDINO often returns normalized boxes [cx, cy, w, h]. 
        # The user's code: `x1, y1, x2, y2 = box` implies it returns xyxy.
        # If it returns normalized, we need to convert. 
        # I'll assume the wrapper handles it or returns xyxy as implied by `x1, y1, x2, y2 = box`.
        
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=False)
        
        # Save cropped window
        x1, y1, x2, y2 = map(int, box) # Ensure ints for indexing
        
        # Clip to image bounds
        h, w, _ = img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        output_filename = os.path.join(args.output_dir, f"window_crop_{i}.png")
        cv2.imwrite(output_filename, crop)
        print(f"Saved {output_filename}")

        # Optional: Save mask overlay
        # ...

if __name__ == "__main__":
    main()
