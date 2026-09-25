https://fh3s4fcd-8000.asse.devtunnels.ms/

## Forecasting models

The forecast endpoint keeps the six existing baseline models and compares them
with an optional GRU neural model using rolling holdout backtesting. The GRU is
disabled by default so the lightweight Docker image and CI remain fast.

To enable it locally, install the optional dependency and set the feature flag:

```powershell
python -m pip install -e ".[forecast-gru]"
$env:MEAL_HISTORY_ENABLE_GRU = "1"
$env:DATA_PATH = "$PWD\data\meal_history_clean.csv"
python -m meal_history.web
```

When enabled and at least 30 daily observations are available, `GRU` appears
in `model_leaderboard`. Automatic model selection uses the lowest holdout MAPE;
if GRU is unavailable or there is insufficient history, the existing baseline
selection is used unchanged.
