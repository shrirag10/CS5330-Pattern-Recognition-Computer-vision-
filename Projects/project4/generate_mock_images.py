import cv2
import numpy as np
import os

def create_mock_calibration_images():
    # Read the base checkerboard
    img = cv2.imread("checkerboard.png")
    if img is None:
        print("Error: checkerboard.png not found. Run generate_checkerboard.py first.")
        return

    h, w = img.shape[:2]
    
    # Original corners of the chessboard grid (inside the white border)
    # The grid is from x=100 to 1100, y=100 to 800
    src_pts = np.float32([
        [100, 100],
        [1100, 100],
        [1100, 800],
        [100, 800]
    ])
    
    # Create output directory
    os.makedirs("data/mock_calibration", exist_ok=True)
    
    # Define 5 different perspective transformations
    destinations = [
        # 1. Tilt down-right
        [[150, 150], [1050, 120], [1080, 780], [120, 820]],
        # 2. Tilt down-left
        [[120, 120], [1000, 180], [1050, 820], [150, 780]],
        # 3. Tilt forward (top squished, bottom wider)
        [[250, 200], [950, 200], [1120, 820], [80, 820]],
        # 4. Tilt backward (top wide, bottom squished)
        [[80, 150], [1120, 150], [950, 750], [250, 750]],
        # 5. Rotate and translate
        [[180, 100], [1120, 200], [1020, 850], [80, 750]]
    ]
    
    # Create desk background (brown wooden grain)
    np.random.seed(42)  # For reproducible texture
    bg_base = np.zeros((h, w, 3), dtype=np.uint8)
    bg_base[:, :] = [35, 55, 75]  # Brown BGR
    # Add grain/stripes
    for y in range(0, h):
        factor = 1.0 + 0.15 * np.sin(y / 8.0) + 0.05 * np.sin(y / 2.0)
        bg_base[y, :, :] = np.clip(bg_base[y, :, :].astype(float) * factor, 0, 255).astype(np.uint8)
    # Add fine texture noise
    noise_bg = np.random.normal(0, 4, (h, w, 3)).astype(np.int16)
    bg_base = np.clip(bg_base.astype(np.int16) + noise_bg, 0, 255).astype(np.uint8)

    # White sheet mask in source image
    paper_mask = np.ones((h, w), dtype=np.uint8) * 255

    for idx, dst in enumerate(destinations):
        dst_pts = np.float32(dst)
        # Compute perspective matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp the checkerboard sheet and the mask
        warped_paper = cv2.warpPerspective(img, M, (w, h), borderValue=0)
        warped_mask = cv2.warpPerspective(paper_mask, M, (w, h), borderValue=0)
        
        # Blend sheet onto desk background
        mask_3d = np.repeat(warped_mask[:, :, np.newaxis], 3, axis=2)
        combined = np.where(mask_3d > 0, warped_paper, bg_base)
        
        # Simulate uneven lighting (vignetting + directional spotlight)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        # Center of light shift slightly per frame to simulate moving camera/light
        cx = 300 + (idx * 100)
        cy = 200 + (idx * 50)
        dist_sq = (X - cx)**2 + (Y - cy)**2
        light_map = 0.60 + 0.40 * np.exp(-dist_sq / (2 * 550**2))
        combined = (combined * light_map[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
        
        # Simulate slight webcam lens blur
        combined = cv2.GaussianBlur(combined, (5, 5), 0.7)
        
        # Simulate sensor noise
        sensor_noise = np.random.normal(0, 3.5, (h, w, 3)).astype(np.int16)
        combined = np.clip(combined.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)
        
        # Save image
        out_path = f"data/mock_calibration/mock_{idx+1}.png"
        cv2.imwrite(out_path, combined)
        print(f"Generated realistic image: {out_path}")

if __name__ == "__main__":
    create_mock_calibration_images()
