# Dataset Launchers

This folder contains one clean `.sh` launcher for each dataset used in the RCARE-Forecast experiments:

- `ETTh1.sh`
- `ETTh2.sh`
- `appliances_energy.sh`
- `AQShunyi.sh`
- `AQWan.sh`
- `beijing_pm25.sh`
- `CzeLan.sh`
- `exchange_rate.sh`
- `Flight.sh`
- `weather.sh`
- `Wind.sh`
- `ZafNoo.sh`

Default behavior:

```bash
bash scripts/AQShunyi.sh
```

This runs both numeric warm-up and full RCARE training over:

```text
PRED_LENS="96 192 336 720"
RATIOS="0.05 0.10 0.20"
```

Useful overrides:

```bash
RUN_STAGE=numeric bash scripts/AQShunyi.sh
RUN_STAGE=full PRED_LENS="96" RATIOS="0.10" bash scripts/AQShunyi.sh
RUN_STAGE=both BATCH_SIZE=48 EVAL_BATCH_SIZE=64 bash scripts/AQShunyi.sh
PYTHON=/path/to/python bash scripts/AQShunyi.sh
```

If you already have a numeric warm-up checkpoint and want to freeze it in the full RCARE stage:

```bash
NUMERIC_CKPT="checkpoints/<numeric-setting>/checkpoint.pth" RUN_STAGE=full bash scripts/AQShunyi.sh
```

Before running, prepare:

```text
dataset/<dataset>.csv
generated/<dataset>/<dataset>_sl96_pl<pred_len>_text_M_semantic_v1.csv
generated/<dataset>/<dataset>_sl96_pl<pred_len>_semantic_v1_text_features.npz
```
