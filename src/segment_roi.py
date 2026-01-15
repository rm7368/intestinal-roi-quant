"""
Segment ROI into equal arc-length portions along the intestinal axis.
"""

import numpy as np
import json
from pathlib import Path
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def load_roi(roi_path):
    """Load ROI from JSON file."""
    with open(roi_path, 'r') as f:
        roi_data = json.load(f)
    return roi_data


def compute_arc_length(coords):
    """
    Compute cumulative arc length along a path.
    
    Parameters
    ----------
    coords : array (N, 2)
        Path coordinates
        
    Returns
    -------
    arc_length : array (N,)
        Cumulative arc length at each point
    """
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    arc_length = np.concatenate([[0], np.cumsum(distances)])
    return arc_length


def divide_centerline_by_arc_length(centerline, n_segments):
    """
    Divide centerline into n_segments of equal arc length.
    
    Parameters
    ----------
    centerline : array (N, 2)
        Centerline coordinates
    n_segments : int
        Number of segments to divide into
        
    Returns
    -------
    segment_points : array (n_segments+1, 2)
        Coordinates of segment boundaries
    segment_indices : array (n_segments+1,)
        Indices in original centerline closest to boundaries
    """
    # Compute arc length
    arc_length = compute_arc_length(centerline)
    total_length = arc_length[-1]
    
    print(f"Total centerline arc length: {total_length:.1f} pixels")
    print(f"Dividing into {n_segments} segments of {total_length/n_segments:.1f} pixels each")
    
    # Find evenly spaced arc length points
    target_lengths = np.linspace(0, total_length, n_segments + 1)
    
    # Find closest points on centerline to target arc lengths
    segment_indices = []
    segment_points = []
    
    for target in target_lengths:
        idx = np.argmin(np.abs(arc_length - target))
        segment_indices.append(idx)
        segment_points.append(centerline[idx])
    
    segment_indices = np.array(segment_indices)
    segment_points = np.array(segment_points)
    
    return segment_points, segment_indices


def create_segment_masks(tube_coords, centerline, segment_indices, image_shape):
    """
    Create binary masks for each segment of the tube.
    
    Parameters
    ----------
    tube_coords : array (M, 2)
        Tube polygon coordinates
    centerline : array (N, 2)
        Centerline coordinates
    segment_indices : array (n_segments+1,)
        Indices marking segment boundaries on centerline
    image_shape : tuple
        Shape of the image (height, width)
        
    Returns
    -------
    masks : list of arrays
        Binary mask for each segment
    """
    from skimage.draw import polygon
    
    n_segments = len(segment_indices) - 1
    masks = []
    
    # Split tube coords into two halves (left and right boundaries)
    n_tube = len(tube_coords)
    left_boundary = tube_coords[:n_tube//2]
    right_boundary = tube_coords[n_tube//2:][::-1]  # Reverse right side
    
    for i in range(n_segments):
        # Get segment boundaries on centerline
        start_idx = segment_indices[i]
        end_idx = segment_indices[i + 1]
        
        # Create polygon for this segment
        # Left boundary for this segment
        left_seg = left_boundary[start_idx:end_idx+1]
        # Right boundary for this segment (reversed)
        right_seg = right_boundary[start_idx:end_idx+1][::-1]
        
        # Combine into closed polygon
        segment_poly = np.vstack([left_seg, right_seg])
        
        # Create mask
        rr, cc = polygon(segment_poly[:, 0], segment_poly[:, 1], shape=image_shape)
        mask = np.zeros(image_shape, dtype=bool)
        mask[rr, cc] = True
        
        masks.append(mask)
        print(f"Segment {i+1}: {mask.sum()} pixels")
    
    return masks


def segment_roi(roi_path, n_segments, output_dir=None):
    """
    Main function to segment ROI into equal arc-length portions.
    
    Parameters
    ----------
    roi_path : str or Path
        Path to ROI JSON file
    n_segments : int
        Number of segments to create
    output_dir : str or Path, optional
        Directory to save segment masks
        
    Returns
    -------
    masks : list of arrays
        Binary mask for each segment
    segment_data : dict
        Metadata about segmentation
    """
    roi_path = Path(roi_path)
    
    # Load ROI
    roi_data = load_roi(roi_path)
    centerline = np.array(roi_data['centerline'])
    tube_coords = np.array(roi_data['tube_coords'])
    image_shape = tuple(roi_data['image_shape'])
    
    print(f"\nLoaded ROI from {roi_path}")
    print(f"Centerline: {len(centerline)} points")
    print(f"Tube boundary: {len(tube_coords)} points")
    
    # Divide centerline into segments
    segment_points, segment_indices = divide_centerline_by_arc_length(
        centerline, n_segments
    )
    
    # Create masks for each segment
    print(f"\nCreating segment masks...")
    masks = create_segment_masks(
        tube_coords, centerline, segment_indices, image_shape
    )
    
    # Prepare output
    segment_data = {
        'n_segments': n_segments,
        'segment_points': segment_points.tolist(),
        'segment_indices': segment_indices.tolist(),
        'roi_path': str(roi_path)
    }
    
    # Save if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save segment data
        with open(output_dir / 'segment_data.json', 'w') as f:
            json.dump(segment_data, f, indent=2)
        
        # Save masks as numpy arrays
        for i, mask in enumerate(masks):
            np.save(output_dir / f'segment_{i+1}_mask.npy', mask)
        
        print(f"\nSaved {n_segments} segment masks to {output_dir}")
    
    return masks, segment_data


def visualize_segments(image_path, roi_path, n_segments, output_path='segments_viz.png'):
    """
    Visualize the segmentation overlaid on the image.
    
    Parameters
    ----------
    image_path : str
        Path to original TIFF image
    roi_path : str
        Path to ROI JSON file
    n_segments : int
        Number of segments
    output_path : str
        Where to save visualization
    """
    import tifffile
    
    # Load image
    image = tifffile.imread(image_path)
    
    # Segment ROI
    masks, segment_data = segment_roi(roi_path, n_segments)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='gray', alpha=0.7)
    
    # Overlay each segment with different color
    colors = plt.cm.tab10(np.linspace(0, 1, n_segments))
    
    for i, mask in enumerate(masks):
        # Create colored overlay
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask, :3] = colors[i, :3]
        overlay[mask, 3] = 0.3  # Alpha
        
        ax.imshow(overlay)
        
        # Label segment
        y, x = np.where(mask)
        cy, cx = y.mean(), x.mean()
        ax.text(cx, cy, f'{i+1}', color='white', fontsize=20, 
                ha='center', va='center', fontweight='bold')
    
    ax.set_title(f'ROI divided into {n_segments} equal arc-length segments')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    
    return fig


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python segment_roi.py <roi.json> <n_segments> [--visualize]")
        print("Example: python segment_roi.py output.roi.json 10 --visualize")
        sys.exit(1)
    
    roi_path = sys.argv[1]
    n_segments = int(sys.argv[2])
    visualize = '--visualize' in sys.argv
    
    # Segment the ROI
    masks, segment_data = segment_roi(roi_path, n_segments, output_dir='segment_output')
    
    # Visualize if requested
    if visualize:
        roi_data = load_roi(roi_path)
        image_path = roi_data['image_path']
        visualize_segments(image_path, roi_path, n_segments)