# Intestinal ROI Quantification Pipeline

Interactive pipeline for segmenting and quantifying intestinal swiss roll fluorescence images.

## Overview

This tool allows you to:
1. Interactively select ROI from large DAPI-stained intestinal swiss rolls
2. Segment the ROI into equal-length portions along the intestinal axis
3. Perform epithelial cell segmentation
4. Export results in FIJI-compatible format

## Installation
```bash
# Clone the repository
git clone https://github.com/rm7368/intestinal-roi-quant.git
cd intestinal-roi-quant

# Create conda environment
conda env create -f environment.yml
conda activate intestinal-roi-quant
```

## Quick Start
```bash
# Run interactive ROI selection
python scripts/run_pipeline.py --input /path/to/image.tif
```

## Usage

See [detailed usage documentation](docs/usage.md).

## Development

This pipeline is under active development in the Zwick Lab at NYU Grossman School of Medicine.
