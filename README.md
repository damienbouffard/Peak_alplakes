# Visualisation of 1D and 3D model output

**Date:** 2025-12-01

**Authors:** Damien Bouffard, Martin Schmid 

## Overview

This repository contains Jupyter Notebook with utilities to visualise 1D lake model output (Simstrat-like `T_out.dat`) and 3D lake model output. Model output can be accessed from Alplakes: https://www.alplakes.eawag.ch

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
or 

```bash
jupyter notebook scripts/3D_visualisation_map.ipynb
```

3. Possible visualisations 1D

- `plot_temperature_heatmap(df, years=None, ...)` — plot temperature heatmap (depth vs time).
- `plot_temperature_at_depth(df, depth, ...)` — extract and plot a time series at the nearest available depth (with aggregation options).
- `plot_temperature_climatology(df, depth, ...)` — compute and plot daily-of-year climatology (mean, std, min, max) with optional baseline-year overlay.
- `compare_heatmaps(path1, path2, ...)` — load two site outputs and plot their difference heatmap (restricted to shallowest system).
- `plot_two_sites_at_depth(path1, path2, depth, ...)` — extract and plot aligned time series from two sites at a given depth.

3. Possible visualisations 3D

- `plot_alplakes_pcolormesh(data, ...)` — plot a map of temperature and current for a given lake at a given time and depth .
- `plot_temperature_timeseries(data, ...)` — plot a time serie of temperature for a given lake at a given location (x,y,z) over a given period.
- `plot_alplakes_transect(data,...)` — plot a transect for a given lake at a given time. Transect is defined with lat, lon coordinates
- `plot_alplakes_transect_timeseries(data, ...` — plot a transect for a given lake over a given period. Transect is defined with lat, lon coordinates.


