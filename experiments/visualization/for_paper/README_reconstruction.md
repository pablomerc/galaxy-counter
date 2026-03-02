# Reconstruction Visualization Scripts

This directory contains scripts for generating and visualizing reconstruction results from pretrained neighbors models.

## Files

- **`reconstruction_pretrained_neighbor.py`**: Main script that loads a pretrained model, processes validation data, generates reconstructions, and saves results
- **`replot_reconstruction.py`**: Fast replotting script that reads saved data and regenerates plots for specific examples
- **`README_reconstruction.md`**: This file

## Quick Start

### 1. Generate Reconstructions

First, run the main script to load the model and generate reconstructions:

```bash
cd /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/visualization_scripts/for_paper
python reconstruction_pretrained_neighbor.py
```

This will:
- Load the pretrained model from checkpoint
- Process 4 validation examples (configurable via `NUM_EXAMPLES`)
- Generate 5 reconstruction samples per example (configurable via `NUM_SAMPLES`)
- Create a reconstruction plot showing: `SameGal | SameIns (1st) | Target | Sample1 | ... | SampleN | Mean`
- Save all data to `reconstruction_outputs/reconstruction_data.h5` for quick replotting

**Outputs:**
- `reconstruction_outputs/reconstruction_plot.png`: Main reconstruction visualization
- `reconstruction_outputs/reconstruction_data.h5`: Saved data including ALL instrument pairs, samples, and metadata

### 2. Quick Replotting

After generating the data, you can quickly replot specific examples without rerunning the model:

```bash
# Show information about available examples
python replot_reconstruction.py --info

# Plot a single example (index 0)
python replot_reconstruction.py --index 0

# Plot multiple specific examples
python replot_reconstruction.py --indices 0 1 2

# Plot all examples
python replot_reconstruction.py --all

# Use custom output path
python replot_reconstruction.py --index 0 --output my_plot.png
```

## Configuration

Edit the configuration section in `reconstruction_pretrained_neighbor.py` to customize:

### Model Checkpoint

```python
CHECKPOINT_PATH = "/path/to/your/checkpoint.ckpt"
```

Available checkpoints (from `load_pretrained_model_neighbors.py`):
- **64 no geo**: `/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/n8szckjq/checkpoints/latest-step=step=56000.ckpt`
- **64 geo**: `/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/lr451mnx/checkpoints/latest-step=step=53000.ckpt`
- **16 no geo** (default): `/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt`

### Data and Output

```python
# Validation data path
VAL_DATA_PATH = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/val_neighbors.vds"

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'reconstruction_outputs'

# Number of examples to process
NUM_EXAMPLES = 4  # Tune this later

# Number of samples to generate per example
NUM_SAMPLES = 5
```

## What Gets Saved

The HDF5 data file (`reconstruction_data.h5`) contains:

### Image Data
- `targets`: Ground truth target images (batch_size, C, H, W)
- `samegals`: Same galaxy conditioning images (batch_size, C, H, W)
- `sameins`: **ALL** same instrument neighbors (batch_size, k, C, H, W)
- `masks`: Valid neighbor masks (batch_size, k)
- `samples`: Generated reconstruction samples (batch_size, num_samples, C, H, W)
- `mean_samples`: Mean of all samples (batch_size, C, H, W)

### Metadata
- `anchor_surveys`: Survey source for each example (HSC or Legacy)
- `indices`: Original dataset indices
- `num_same_instrument`: Number of valid neighbors for each example

### Attributes
- `batch_size`, `num_samples`, `image_channels`, `image_height`, `image_width`, `num_neighbors`

## Survey Information

Each reconstruction plot shows which survey the target comes from:
- **HSC**: Displayed with color `#e8c4a0` (light orange/beige)
- **Legacy**: Displayed with color `#8eb8e8` (light blue)

The survey label appears on the left side of each row in the plot.

## Reconstruction Layout

The plots show the reconstruction process:

1. **SameGal**: Same galaxy (different instrument) - conditioning input
2. **SameIns (1st)**: First same instrument neighbor - conditioning input
3. **Target**: Ground truth target to reconstruct
4. **Sample 1-N**: Multiple stochastic reconstruction samples
5. **Mean**: Average of all samples

All images in a row are scaled using the same colormap based on the target image's min/max values (row-scaled visualization).

## Example Use Cases

### Research Paper Figures

1. Generate reconstructions for multiple validation examples:
   ```bash
   # Edit NUM_EXAMPLES in reconstruction_pretrained_neighbor.py
   python reconstruction_pretrained_neighbor.py
   ```

2. Select best examples for paper:
   ```bash
   python replot_reconstruction.py --info  # See MSE for each example
   python replot_reconstruction.py --indices 0 2  # Plot best two examples
   ```

### Exploring Specific Cases

1. Find interesting cases (e.g., high/low MSE, specific survey):
   ```bash
   python replot_reconstruction.py --info
   ```

2. Visualize specific examples:
   ```bash
   python replot_reconstruction.py --index 3
   ```

### Batch Processing

Generate plots for all examples at once:
```bash
python replot_reconstruction.py --all
```

## Tips

1. **Memory**: The main script loads the model and generates samples, which requires GPU memory. The replotting script only loads saved data and is very fast.

2. **Data Size**: The HDF5 file includes ALL same-instrument neighbors (not just the first one shown in the plot), so you can later explore different neighbor combinations if needed.

3. **Customization**: Both scripts use the same `_row_scale_rgb` function for visualization consistency. Modify this function if you want different color scaling.

4. **Multiple Runs**: You can run the main script with different checkpoints or validation data, just change the output directory to avoid overwriting results.

## Troubleshooting

### Model loading fails
- Check that the checkpoint path is correct
- Ensure you have enough GPU/CPU memory
- The script will automatically fall back to CPU if GPU is unavailable

### Data file not found
- Run `reconstruction_pretrained_neighbor.py` first to generate the data
- Check that `VAL_DATA_PATH` points to the correct validation data file

### Out of memory
- Reduce `NUM_EXAMPLES` or `NUM_SAMPLES`
- Close other GPU-intensive processes
- Use CPU by setting `CUDA_VISIBLE_DEVICES=""`

## Contact

For questions or issues, contact the author or refer to the parent training script documentation.
