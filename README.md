# Visualisation of 1D and 3D model output

**Date:** 2025-12-01

**Authors:** Damien Bouffard, Martin Schmid 

## Overview

This repository contains a Jupyter Notebook (`scripts/1D_visualisation_simstrat.ipynb`) with utilities to visualise 1D lake model output (Simstrat-like `T_out.dat`). The notebook provides functions to:

- Plot a temperature heatmap (depth vs time).
- Extract and plot a time series at the nearest available depth (with aggregation options).
- Compute and plot daily-of-year climatologies (mean, std, min, max) with optional baseline-year overlay.
- Compare two site outputs by plotting their difference heatmap (limited to the shallower system's depth range).
- Plot aligned time series from two sites at a specified depth.


## Quick start

1. Create and activate a Python environment (example using conda):

```bash
conda create -n peak_alplakes python=3.10 -y
conda activate peak_alplakes
pip install -r requirements.txt
```

2. Open the notebook:

```bash
jupyter notebook scripts/1D_visualisation_simstrat.ipynb
```

3. Example (run inside the notebook):

```python
# load data (adjust path in the notebook if needed)
# plot heatmap for 2024
plot_temperature_heatmap(df, years=2024)

# plot temperature at ~10 m (monthly aggregated)
plot_temperature_at_depth(df, depth=10, agg='monthly')

# compute and plot climatology
ax, clim = plot_temperature_climatology(df, depth=0, period=(1981,2024), baseline_year=2012)

# compare two sites (difference heatmap)
compare_heatmaps('data/Geneva', 'data/Upper_Lugano', years=(2000,2020))
```

## Running notebook headless (execute & save outputs)

```bash
./run_notebook.sh scripts/1D_visualisation_simstrat.ipynb
```




