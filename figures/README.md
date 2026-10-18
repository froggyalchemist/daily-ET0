# Figures

#TODO: Update figure title and caption when I know the final figure captions

This folder contains the **Python notebooks (`.ipynb`) used to create the figures**, and the resulting figures in **PNG** (`png/`) and **PDF** (`pdf/`) format.

Notebooks are named after the figure they produce (e.g. `Figure2.ipynb`, `FigureS3.ipynb`). Notebooks ending in `_left`, `_right`, `_upper`, `_top` or `_bottom` create individual panels, and `*_merge_panels.ipynb` notebooks assemble them into the final figure. They read the derived data described in the [main README](../README.md) and use the helpers in [`plot_utils.py`](../CMIP6/plot_utils.py).

Click a title below to expand the figure.


## Main figures

<details>
<summary><b>Figure 1 – Thirstwave detection, metrics, and global patterns</b></summary>

<br>

**A–B.** Example of thirstwave detection at one grid cell and schematic of the metrics (intensity, duration). **C–H.** Global maps of frequency, intensity and duration, and their trends.

<img src="png/Figure1_full.png" alt="Figure 1" width="100%">

</details>

<details>
<summary><b>Figure 2 – Radiative contribution to thirstwave anomalies</b></summary>

<br>

**A–B.** Fractional contribution of the radiative term (ΔETos<sub>Rad</sub>) and its trend. **C–D.** Aridity-index percentile and dependence of the radiative contribution and its trend on aridity.

<img src="png/Figure2_combined.png" alt="Figure 2" width="100%">

</details>

<details>
<summary><b>Figure 3 – Projected changes in thirstwave characteristics</b></summary>

<br>

**A, C, E, G.** Time series (1980–2100) of frequency, intensity, duration and radiative contribution for historical, SSP1-2.6 to SSP5-8.5 and ERA5. **B, D, F, H.** Corresponding maps of change.

<img src="png/Figure3.png" alt="Figure 3" width="100%">

</details>

<details>
<summary><b>Figure 4 – Crop exposure to thirstwaves</b></summary>

<br>

**A–D.** Crop exposure anomaly maps for SSP1-2.6 to SSP5-8.5. **E.** Crop exposure anomaly by socioeconomic region and scenario, regions ordered by 1981–2000 crop fraction.

<img src="png/Figure4_combined.png" alt="Figure 4" width="100%">

</details>



## Supplementary figures

<details>
<summary><b>Figure S3 – 90th-percentile ETos threshold per model</b></summary>

<br>

Growing-season 90th-percentile threshold (1981–2000) for each of the 8 GCMs.

<img src="png/FigureS3.png" alt="Figure S3" width="100%">

</details>

<details>
<summary><b>Figure S4 – ETos climatology per model</b></summary>

<br>

Growing-season ETos climatology (1981–2000) for each of the 8 GCMs.

<img src="png/FigureS4.png" alt="Figure S4" width="100%">

</details>

<details>
<summary><b>Figure S5 – Change in thirstwave frequency</b></summary>

<br>

Change in frequency, 2071–2100 minus 1981–2000 (%), for each SSP.

<img src="png/FigureS5.png" alt="Figure S5" width="100%">

</details>

<details>
<summary><b>Figure S6 – Change in thirstwave intensity</b></summary>

<br>

Change in intensity, 2071–2100 minus 1981–2000 (%), for each SSP.

<img src="png/FigureS6.png" alt="Figure S6" width="100%">

</details>

<details>
<summary><b>Figure S7 – Change in thirstwave duration</b></summary>

<br>

Change in duration, 2071–2100 minus 1981–2000 (%), for each SSP.

<img src="png/FigureS7.png" alt="Figure S7" width="100%">

</details>

<details>
<summary><b>Figure S8 – Cropland area</b></summary>

<br>

Cropland area in 2071–2100 (% of grid cell area) for each SSP.

<img src="png/FigureS8.png" alt="Figure S8" width="100%">

</details>

<details>
<summary><b>Figure S9 – Definition of the 30 regions</b></summary>

<br>

Socioeconomic regions used in Figure 4E.

<img src="png/FigureS9.png" alt="Figure S9" width="100%">

</details>
