# ERA5
This folder contains the scripts used to to calculate ETos from ERA5 data (`daily_ETos_calculation` ) and to perform the thirstwave detection and other analyses.

## Daily $ETos$ calculation
Run the files in `daily_ETos_calculation`  first (a, b, c, then etos).

#todo specify what a, b and c are.

## Thirstwave detection
Then, analysis. must be run in a specific order because there are dependencies:

Order | File | output | description
1 tx01_90.py --> "tx01_90_c.nc" (90th percentile)
2 thirstwave_detection.py --> "valid_event_days_1981_2023_new.nc"
3 Analysis_TW_growing_season.ipynb --> "thirstwave_metrics_growing_season.nc"
4 Analysis_TW_trend_growing_season.ipynb --> "thirstwave_trend_growing_season.nc"
