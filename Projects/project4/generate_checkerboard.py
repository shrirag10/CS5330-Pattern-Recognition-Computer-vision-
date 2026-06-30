import numpy as np
import cv2

def generate_checkerboard(cols=10, rows=7, square_size=100, border=100):
    # Determine chessboard grid size
    grid_width = cols * square_size
    grid_height = rows * square_size
    
    # Create the chessboard grid image (0 = black, 255 = white)
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            # Alternating black and white squares
            if (r + c) % 2 == 1:
                y0, x0 = r * square_size, c * square_size
                grid[y0:y0+square_size, x0:x0+square_size] = 255
                
    # Add a quiet zone (white border) around the grid
    img = np.ones((grid_height + 2 * border, grid_width + 2 * border), dtype=np.uint8) * 255
    img[border:border+grid_height, border:border+grid_width] = grid
    
    # Save the target image
    cv2.imwrite("checkerboard.png", img)
    print("Saved checkerboard.png with 10x7 squares (9x6 internal corners).")

if __name__ == "__main__":
    generate_checkerboard()
