import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import sys

def visualize_rgb(image_path):
    # Load image
    img = iio.imread(image_path)

    # If it has an alpha channel, strip it
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = img[:, :, :3]
        print("Alpha channel detected and ignored for visualization.")
    else:
        rgb = img

    # Show image
    plt.imshow(rgb)
    plt.title("RGB Content")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_rgb.py path/to/image.png")
    else:
        visualize_rgb(sys.argv[1])
