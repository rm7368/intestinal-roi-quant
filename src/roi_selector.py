"""
Interactive ROI selection for intestinal swiss roll images using napari.
"""

import numpy as np
import napari
from pathlib import Path
import tifffile
import json
from scipy.interpolate import splprep, splev, interp1d

def expand_centerline_variable_width(centerline, widths):
    """
    Expand centerline to tube with variable width.
    
    Parameters
    ----------
    centerline : array (N, 2)
        Centerline coordinates
    widths : array (N,)
        Half-width at each centerline point
    """
    # Compute tangents
    tangents = np.diff(centerline, axis=0, prepend=centerline[0:1])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-10)
    
    # Perpendiculars
    perps = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    
    # Create boundaries
    left = centerline - perps * widths[:, None]
    right = centerline + perps * widths[:, None]
    
    # Close the polygon
    tube = np.vstack([left, right[::-1]])
    return tube


class CenterlineDrawer:
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.image = tifffile.imread(image_path)
        self.viewer = None
        self.centerline = None
        self.width_default = 50  # Default half-width
        
    def draw(self):
        self.viewer = napari.Viewer()
        
        self.viewer.add_image(
            self.image,
            name='DAPI',
            colormap='gray',
            contrast_limits=[np.percentile(self.image, 1), 
                           np.percentile(self.image, 99)]
        )
        
        centerline_layer = self.viewer.add_shapes(
            name='Draw centerline (use path tool)',
            edge_color='yellow',
            edge_width=2,
            shape_type='polygon'
        )
        
        print("\n" + "="*60)
        print("STEP 1: Draw centerline")
        print("="*60)
        print("1. Select 'add path' tool")
        print("2. Click points along center of intestine")
        print("3. Press ESC when done, then close window")
        print("="*60 + "\n")
        
        napari.run()
        
        if len(centerline_layer.data) > 0:
            self.centerline = centerline_layer.data[0]
            print(f"Centerline: {len(self.centerline)} points")
            return self.centerline
        return None
    
    def generate_tube(self, width=50):
        """Generate tube with constant width."""
        if self.centerline is None:
            raise ValueError("Draw centerline first!")
        
        widths = np.full(len(self.centerline), width)
        tube = expand_centerline_variable_width(self.centerline, widths)
        return tube


class TubeRefiner:
    def __init__(self, image_path, tube_coords):
        self.image_path = Path(image_path)
        self.image = tifffile.imread(image_path)
        self.tube_coords = tube_coords
        self.viewer = None
        
    def refine(self):
        self.viewer = napari.Viewer()
        
        self.viewer.add_image(
            self.image,
            name='DAPI',
            colormap='gray',
            contrast_limits=[np.percentile(self.image, 1), 
                           np.percentile(self.image, 99)]
        )
        
        tube_layer = self.viewer.add_shapes(
            [self.tube_coords],
            shape_type='polygon',
            edge_color='cyan',
            face_color=[0, 1, 1, 0.1],
            edge_width=2,
            name='Tube ROI (drag vertices to adjust)'
        )
        
        print("\n" + "="*60)
        print("STEP 2: Refine tube boundaries")
        print("="*60)
        print("1. Switch to 'select vertices' mode")
        print("2. Drag individual vertices to adjust boundaries")
        print("3. Close window when satisfied")
        print("="*60 + "\n")
        
        napari.run()
        
        if len(tube_layer.data) > 0:
            refined = tube_layer.data[0]
            print(f"Refined tube: {len(refined)} vertices")
            return refined
        return self.tube_coords


def select_roi_two_step(image_path, output_path=None, initial_width=50):
    """
    Two-step ROI selection process.
    
    Parameters
    ----------
    image_path : str
        Path to TIFF image
    output_path : str, optional
        Where to save ROI
    initial_width : int
        Initial tube half-width in pixels
    """
    # Step 1: Draw centerline
    drawer = CenterlineDrawer(image_path)
    centerline = drawer.draw()
    
    if centerline is None:
        print("No centerline drawn!")
        return None
    
    # Generate initial tube
    tube = drawer.generate_tube(width=initial_width)
    print(f"\nGenerated tube with width={initial_width} pixels")
    
    # Step 2: Refine tube
    refiner = TubeRefiner(image_path, tube)
    refined_tube = refiner.refine()
    
    # Save
    if output_path is None:
        output_path = Path(image_path).with_suffix('.roi.json')
    
    roi_data = {
        'image_path': str(image_path),
        'image_shape': list(drawer.image.shape),
        'centerline': centerline.tolist(),
        'tube_coords': refined_tube.tolist(),
        'initial_width': initial_width
    }
    
    with open(output_path, 'w') as f:
        json.dump(roi_data, f, indent=2)
    
    print(f"\nROI saved to {output_path}")
    return refined_tube


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python roi_selector.py <image.tif> [width]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    select_roi_two_step(image_path, initial_width=width)

