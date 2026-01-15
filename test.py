# Save this as view_segments.py
import napari
import numpy as np
import tifffile

# Load image
image = tifffile.imread("intestinal-roi-quant/examples/test_data/C1-251210_P18_3_Hmgcs2 protein-01_processed_DAPI.tif")

# Load masks
masks = []
for i in range(1, 11):
    mask = np.load(f'segment_output/segment_{i}_mask.npy')
    masks.append(mask)

# Stack masks into a single labeled image
labels = np.zeros(image.shape, dtype=np.uint8)
for i, mask in enumerate(masks):
    labels[mask] = i + 1

# View in napari
viewer = napari.Viewer()
viewer.add_image(image, name='DAPI', colormap='gray')
viewer.add_labels(labels, name='Segments')

print("\nViewing segments in napari!")
print("Each segment has a different color")
print("Close the window when done")

napari.run()