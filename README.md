# Intestinal ROI Quantification Pipeline

Interactive pipeline for segmenting and quantifying intestinal swiss roll fluorescence images.

## Updates

- v1.1.0: Added batch mode. Access in either command using "--batch" flag.
- v1.0.0: First release, manual centerline drawing and edge refinement. One image at a time.

## Overview

This tool allows you to:
1. Create an epithelial cell mask from DAPI-stained intestinal swiss rolls.
2. Edit the ROI of epithelial cell mask.
3. Segment the ROI into equal-length portions along the intestinal axis.
4. Quantify fluorescence channel CTFUs for each probe included.

## Installation
```bash
# Clone the repository
git clone https://github.com/rm7368/intestinal-roi-quant.git
cd intestinal-roi-quant

# Create conda environment
conda env create -f environment.yml
conda activate intestinal-roi-quant
```

## Usage

### Segmentation
```bash
# ROI Segmentation (interactive mode w/ file picker and terminal prompts)
python src/roi_segmentation_pipeline.py --test
# ROI Segmentation (command line mode w/ file and segment arguments)
python src/roi_segmentation_pipeline.py /path/to/dapi.tif --n-segments 30
```

### Quantification
```bash
# Image quantification (interactive mode w/ file picker and terminal prompts)
python src/quantify_segments.py --test
# Image quantification (command line mode w/ file arguments, channels are still prompted)
python src/quantify_segments.py /path/to/roi_output_dir
```

## Outputs

- **segments.npy**: Segmented ROI masks (full resolution)
- **quantification_results.xlsx**: CTFU measurements per segment per channel
- **quantification_plot.png**: Line plots of CTFU measurements

## Requirements

- Python 3.10
- napari, scikit-image, scipy, pandas, openpyxl, seaborn
- See 'environment.yml' for complete list
