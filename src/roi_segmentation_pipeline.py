"""
Intestinal ROI Segmentation Pipeline
Combines all interactive and automated steps for ROI selection
"""

import numpy as np
import json
import napari
import tifffile
from pathlib import Path
from skimage import filters, morphology, transform
from skimage.morphology import binary_erosion, disk, remove_small_objects
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import argparse


class IntestinalROISegmenter:
    def __init__(self, image_path, n_segments=10, downscale=0.25):
        """
        Initialize segmenter.
        
        Parameters
        ----------
        image_path : str or Path
            Path to DAPI TIFF image
        n_segments : int
            Number of segments to divide intestine into
        downscale : float
            Downsample factor (0.25 = 4x smaller for speed)
        """
        self.image_path = Path(image_path)
        self.n_segments = n_segments
        self.downscale = downscale
        
        # Load and potentially downsample image
        print(f"Loading image: {self.image_path}")
        self.image_full = tifffile.imread(self.image_path)
        
        if downscale < 1.0:
            print(f"Downsampling by {downscale}x for processing...")
            self.image = transform.rescale(
                self.image_full, 
                downscale, 
                anti_aliasing=True, 
                preserve_range=True
            ).astype(self.image_full.dtype)
        else:
            self.image = self.image_full
            
        print(f"Processing image shape: {self.image.shape}")
        
        # Data storage
        self.epithelial_mask = None
        self.centerline = None
        self.segments = None
        
    def generate_initial_epithelial_mask(self):
        """Step 1: Generate initial epithelial mask via thresholding."""
        print("\n" + "="*60)
        print("STEP 1: Generating initial epithelial mask")
        print("="*60)
        
        # Mask out super bright Peyer's patches
        very_bright = self.image > np.percentile(self.image, 99.5)
        image_masked = self.image.copy()
        image_masked[very_bright] = 0
        
        # Threshold
        thresh_val = filters.threshold_otsu(image_masked)
        epithelium = image_masked > (thresh_val * 0.5)
        epithelium = remove_small_objects(epithelium, min_size=200)
        epithelium[very_bright] = False
        
        # Refine to boundary
        epithelium_base = binary_erosion(epithelium, disk(5))
        self.epithelial_mask = epithelium & ~epithelium_base
        
        print(f"Initial mask: {self.epithelial_mask.sum()} pixels")
        
    def refine_epithelial_mask(self):
        """Step 2: Interactive refinement of epithelial mask."""
        print("\n" + "="*60)
        print("STEP 2: Refine epithelial mask")
        print("="*60)
        
        # Create painting zone (loose threshold)
        very_bright = self.image > np.percentile(self.image, 99.5)
        image_masked = self.image.copy()
        image_masked[very_bright] = 0
        thresh_val = filters.threshold_otsu(image_masked)
        
        dapi_zone = image_masked > (thresh_val * 0.3)
        dapi_zone = remove_small_objects(dapi_zone, min_size=50)
        dapi_zone[very_bright] = False
        
        # Convert to labels
        epithelial_labels = self.epithelial_mask.astype(np.uint8) * 2
        
        # Interactive refinement
        viewer = napari.Viewer()
        viewer.add_image(self.image, name='DAPI', colormap='gray')
        viewer.add_image(dapi_zone, name='Allowed zone', colormap='blue', opacity=0.2)
        labels_layer = viewer.add_labels(epithelial_labels, name='Paint with label 1 to add')
        
        print("Instructions:")
        print("- Paint with LABEL 1 to add epithelial regions")
        print("- Erase or use label 0 to remove")
        print("- Painting constrained to blue zone")
        print("- Close window when done")
        
        napari.run()
        
        # Save refined mask
        painted_areas = labels_layer.data == 1
        self.epithelial_mask = self.epithelial_mask | (painted_areas & dapi_zone)
        
        print(f"Refined mask: {self.epithelial_mask.sum()} pixels")
        
    def draw_centerline(self):
        """Step 3: Interactive centerline drawing."""
        print("\n" + "="*60)
        print("STEP 3: Draw centerline")
        print("="*60)
        
        viewer = napari.Viewer()
        viewer.add_image(self.image, name='DAPI', colormap='gray')
        viewer.add_image(self.epithelial_mask, name='Epithelial cells', 
                        colormap='green', opacity=0.5)
        
        centerline_layer = viewer.add_shapes(
            name='Draw centerline (polygon tool)',
            edge_color='yellow',
            edge_width=2,
            shape_type='polygon'
        )
        
        print("Instructions:")
        print("- Use polygon tool")
        print("- Click along epithelial-mesenchymal boundary")
        print("- Trace spiral from outside to inside")
        print("- Press ESC when done, close window")
        
        napari.run()
        
        if len(centerline_layer.data) > 0:
            self.centerline = centerline_layer.data[0]
            
            # Calculate arc length
            distances = np.sqrt(np.sum(np.diff(self.centerline, axis=0)**2, axis=1))
            arc_length = np.concatenate([[0], np.cumsum(distances)])
            total_length = arc_length[-1]
            
            print(f"Centerline: {len(self.centerline)} points")
            print(f"Total arc length: {total_length:.1f} pixels")
        else:
            raise ValueError("No centerline drawn!")
            
    def generate_segments(self):
        """Step 4: Automated segment generation (optimized)."""
        print("\n" + "="*60)
        print("STEP 4: Generating segments")
        print("="*60)
        
        # Smooth centerline
        tck, u = splprep([self.centerline[:, 0], self.centerline[:, 1]], s=5, k=3)
        u_fine = np.linspace(0, 1, len(self.centerline) * 3)
        centerline_smooth = np.array(splev(u_fine, tck)).T
        
        # Calculate arc length
        distances = np.sqrt(np.sum(np.diff(centerline_smooth, axis=0)**2, axis=1))
        arc_length = np.concatenate([[0], np.cumsum(distances)])
        total_length = arc_length[-1]
        
        # Divide into segments
        segment_boundaries = np.linspace(0, total_length, self.n_segments + 1)
        centerline_segments = np.digitize(arc_length, segment_boundaries)
        centerline_segments = np.clip(centerline_segments, 1, self.n_segments)
        
        # Pre-compute tangents and their angles
        tangents = np.diff(centerline_smooth, axis=0, append=centerline_smooth[-1:])
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / (tangent_norms + 1e-10)
        
        # Pre-compute tangent angles (this is the direction "forward" along spiral)
        tangent_angles = np.arctan2(tangents[:, 1], tangents[:, 0])
        
        print(f"Assigning {self.epithelial_mask.sum()} pixels to {self.n_segments} segments...")
        
        # Build KD-tree
        tree = cKDTree(centerline_smooth)
        epi_coords = np.column_stack(np.where(self.epithelial_mask))
        
        self.segments = np.zeros(self.epithelial_mask.shape, dtype=np.uint8)
        
        # Angular range for valid assignment
        angle_min = -np.pi
        angle_max = -np.pi/4
        
        # Process in larger batches with vectorized operations
        batch_size = 50000
        n_batches = (len(epi_coords) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(epi_coords))
            batch_coords = epi_coords[start:end]
            
            # Query k nearest neighbors for entire batch at once
            batch_distances, batch_indices = tree.query(batch_coords, k=10, workers=-1)
            
            # Vectorized computation for entire batch
            for i in range(len(batch_coords)):
                y, x = batch_coords[i]
                pixel = np.array([y, x])
                
                # Get valid neighbors (within distance threshold)
                valid_mask = batch_distances[i] <= 300
                valid_indices = batch_indices[i][valid_mask]
                valid_distances = batch_distances[i][valid_mask]
                
                if len(valid_indices) == 0:
                    continue
                
                # Vectorized angle computation for all valid neighbors
                cl_points = centerline_smooth[valid_indices]  # Shape: (n_valid, 2)
                to_pixels = pixel - cl_points  # Shape: (n_valid, 2)
                
                # Compute angles to pixel from each centerline point
                pixel_angles = np.arctan2(to_pixels[:, 1], to_pixels[:, 0])
                
                # Get tangent angles for these centerline points
                cl_tangent_angles = tangent_angles[valid_indices]
                
                # Relative angles (normalized to [-π, π])
                relative_angles = pixel_angles - cl_tangent_angles
                relative_angles = np.arctan2(np.sin(relative_angles), np.cos(relative_angles))
                
                # Find points in valid angular range
                in_range = (relative_angles >= angle_min) & (relative_angles <= angle_max)
                
                if not np.any(in_range):
                    continue
                
                # Among valid points, pick the closest one
                valid_in_range = np.where(in_range)[0]
                closest_idx = valid_in_range[np.argmin(valid_distances[in_range])]
                
                # Assign to segment
                assigned_cl_idx = valid_indices[closest_idx]
                self.segments[y, x] = centerline_segments[assigned_cl_idx]
            
            if (batch_idx + 1) % 2 == 0 or batch_idx == n_batches - 1:
                print(f"  Processed {end}/{len(epi_coords)} pixels...")
        
        print(f"Assigned {(self.segments > 0).sum()} pixels")
    def refine_segments(self):
        """Step 5: Interactive segment boundary refinement."""
        print("\n" + "="*60)
        print("STEP 5: Refine segment boundaries")
        print("="*60)
        
        viewer = napari.Viewer()
        viewer.add_image(self.image, name='DAPI', colormap='gray')
        viewer.add_image(self.epithelial_mask, name='Allowed zone', 
                        colormap='blue', opacity=0.2)
        
        # Create separate layer for each segment
        segment_layers = []
        for seg_num in range(1, self.n_segments + 1):
            seg_mask = (self.segments == seg_num).astype(np.uint8)
            layer = viewer.add_labels(seg_mask, name=f'Segment {seg_num}', opacity=0.5)
            segment_layers.append(layer)
            print(f"  Segment {seg_num}: {seg_mask.sum()} pixels")
        
        print("\nInstructions:")
        print("- Toggle segment visibility (eye icons)")
        print("- Paint with label 1 to ADD pixels")
        print("- Erase or paint with 0 to REMOVE")
        print("- Close window when done")
        
        napari.run()
        
        # Combine refined segments
        self.segments = np.zeros(self.segments.shape, dtype=np.uint8)
        for seg_num, layer in enumerate(segment_layers, start=1):
            seg_mask = (layer.data > 0) & self.epithelial_mask
            self.segments[seg_mask] = seg_num
        
        print(f"\nFinal segment counts:")
        for seg_num in range(1, self.n_segments + 1):
            count = (self.segments == seg_num).sum()
            print(f"  Segment {seg_num}: {count} pixels")
            
    def save_results(self, output_dir=None):
        """Save all results to disk, upscaling to full resolution if needed."""
        if output_dir is None:
            output_dir = self.image_path.parent / f"{self.image_path.stem}_roi_output"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}")
        
        # Check if we need to upscale
        if self.downscale < 1.0:
            print(f"Upscaling masks from {self.image.shape} to {self.image_full.shape}...")
            
            from skimage.transform import resize
            
            # Upscale epithelial mask (use nearest neighbor to preserve binary)
            epithelial_mask_full = resize(
                self.epithelial_mask.astype(float),
                self.image_full.shape,
                order=0,  # Nearest neighbor
                preserve_range=True,
                anti_aliasing=False
            ).astype(bool)
            
            # Upscale segments (use nearest neighbor to preserve integer labels)
            segments_full = resize(
                self.segments.astype(float),
                self.image_full.shape,
                order=0,  # Nearest neighbor
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.uint8)
            
            # Upscale centerline coordinates
            scale_factor = 1.0 / self.downscale
            centerline_full = self.centerline * scale_factor if self.centerline is not None else None
            
            print(f"  Epithelial mask: {epithelial_mask_full.sum()} pixels")
            print(f"  Segments: {(segments_full > 0).sum()} pixels")
            
            # Save full-resolution versions
            np.save(output_dir / 'epithelial_mask.npy', epithelial_mask_full)
            np.save(output_dir / 'segments.npy', segments_full)
            
            # Also save downscaled versions for reference
            np.save(output_dir / 'epithelial_mask_downscaled.npy', self.epithelial_mask)
            np.save(output_dir / 'segments_downscaled.npy', self.segments)
            
        else:
            # Already full resolution, save as-is
            epithelial_mask_full = self.epithelial_mask
            segments_full = self.segments
            centerline_full = self.centerline
            
            np.save(output_dir / 'epithelial_mask.npy', epithelial_mask_full)
            np.save(output_dir / 'segments.npy', segments_full)
        
        # Save metadata
        metadata = {
            'image_path': str(self.image_path),
            'processing_shape': list(self.image.shape),
            'full_image_shape': list(self.image_full.shape),
            'downscale': self.downscale,
            'n_segments': self.n_segments,
            'centerline': centerline_full.tolist() if centerline_full is not None else None,
            'upscaled': self.downscale < 1.0
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Saved:")
        print(f"  - epithelial_mask.npy (full resolution)")
        print(f"  - segments.npy (full resolution)")
        if self.downscale < 1.0:
            print(f"  - *_downscaled.npy (processing resolution)")
        print(f"  - metadata.json")
        
        return output_dir
    
    def run_full_pipeline(self):
        """Execute all steps in sequence."""
        print("\n" + "="*60)
        print("INTESTINAL ROI SEGMENTATION PIPELINE")
        print("="*60)
        print(f"Image: {self.image_path.name}")
        print(f"Segments: {self.n_segments}")
        print(f"Downscale: {self.downscale}")
        
        self.generate_initial_epithelial_mask()
        self.refine_epithelial_mask()
        self.draw_centerline()
        self.generate_segments()
        self.refine_segments()
        output_dir = self.save_results()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        
        return output_dir


def select_file_gui(title="Select file", filetypes=[("TIFF files", "*.tif *.tiff")]):
    """Open GUI file picker."""
    from tkinter import Tk, filedialog
    
    root = Tk()
    root.withdraw()  # Hide main window
    root.attributes('-topmost', True)  # Bring to front
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    
    root.destroy()
    return file_path


def main():
    """Main entry point with test mode support."""
    parser = argparse.ArgumentParser(
        description='Intestinal ROI Segmentation Pipeline'
    )
    parser.add_argument('image_path', nargs='?', help='Path to DAPI image')
    parser.add_argument('--n-segments', type=int, default=10, 
                       help='Number of segments (default: 10)')
    parser.add_argument('--downscale', type=float, default=1.0,
                       help='Downscale factor (default: 1.0)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: interactive file selection')
    
    args = parser.parse_args()
    
    # Test mode: interactive file selection
    if args.test:
        print("\n" + "="*60)
        print("TEST MODE: Interactive File Selection")
        print("="*60)
        
        print("\nSelect DAPI image...")
        dapi_path = select_file_gui("Select DAPI image")
        
        if not dapi_path:
            print("No file selected!")
            return
        
        print(f"Selected: {dapi_path}")
        
        # Get number of segments
        n_segments = input(f"\nNumber of segments [default: 10]: ").strip()
        n_segments = int(n_segments) if n_segments else 10
        
        # Run pipeline
        segmenter = IntestinalROISegmenter(
            dapi_path, 
            n_segments=n_segments,
            downscale=args.downscale
        )
        segmenter.run_full_pipeline()
        
    # Batch mode: command line arguments
    elif args.image_path:
        segmenter = IntestinalROISegmenter(
            args.image_path,
            n_segments=args.n_segments,
            downscale=args.downscale
        )
        segmenter.run_full_pipeline()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()