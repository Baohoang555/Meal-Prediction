"""Optional neural forecasting models.

The GRU dependency is deliberately lazy and optional so the API remains
lightweight in the default Docker and CI installations.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass


MIN_GRU_OBSERVATIONS = 30
GRU_LOOKBACK = 14


@dataclass(frozen=True)
class GRUResult:
    prediction: int
    actual: list[float]
    predicted: list[float]
    technical: dict[str, int | float | str]


def gru_is_enabled() -> bool:
    return (
        os.getenv("MEAL_HISTORY_ENABLE_GRU", "").casefold()
        in {"1", "true", "yes", "on"}
        and importlib.util.find_spec("torch") is not None
    )


def gru_forecast(values: list[float], *, horizon: int = 1) -> GRUResult | None:
    """Train a small deterministic GRU and forecast one or more next values.

    Returns ``None`` when GRU support is disabled or the history is too short.
    The caller can then keep the established baseline-only behavior.
    """
    if not gru_is_enabled() or len(values) < MIN_GRU_OBSERVATIONS:
        return None

    import torch
    from torch import nn

    class GRUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=12, batch_first=True)
            self.output = nn.Linear(12, 1)

        def forward(self, batch):
            result, _ = self.gru(batch)
            return self.output(result[:, -1, :])

    holdout_size = max(1, min(14, len(values) // 5))
    train_values = values[:-holdout_size]
    training_values = train_values if len(train_values) >= MIN_GRU_OBSERVATIONS else values
    predict, technical = _fit_gru(training_values, GRUNet, torch, nn)
    validation = []
    if len(train_values) >= MIN_GRU_OBSERVATIONS:
        validation_history = train_values[:]
        for actual_value in values[-holdout_size:]:
            validation.append(predict(validation_history))
            validation_history.append(actual_value)
    model_predictions = []
    rolling = values[:]
    for _ in range(horizon):
        next_value = predict(rolling)
        model_predictions.append(next_value)
        rolling.append(next_value)
    return GRUResult(
        prediction=round(model_predictions[-1]),
        actual=values[-holdout_size:],
        predicted=validation,
        technical={
            **technical,
            "training_samples": len(training_values),
            "validation_samples": len(validation),
        },
    )


def _fit_gru(values, network, torch, nn):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    lookback = min(GRU_LOOKBACK, len(values) // 3)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = variance**0.5 or 1.0
    normalized = [(value - mean) / scale for value in values]
    inputs = [normalized[index - lookback : index] for index in range(lookback, len(normalized))]
    targets = [normalized[index] for index in range(lookback, len(normalized))]
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    x_train = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    model.train()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_function(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    def predict(sequence):
        model.eval()
        recent = [(value - mean) / scale for value in sequence[-lookback:]]
        batch = torch.tensor(recent, dtype=torch.float32).view(1, lookback, 1)
        with torch.no_grad():
            normalized_prediction = float(model(batch).item())
        return max(0.0, normalized_prediction * scale + mean)

    return predict, {
        "lookback": lookback,
        "hidden_size": 12,
        "epochs": 80,
        "learning_rate": 0.01,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


ADVANCED_MODEL_SPECS = (
    ("Random Forest", "sklearn", "forecast-ml"),
    ("Gradient Boosting", "sklearn", "forecast-ml"),
    ("XGBoost", "xgboost", "forecast-tree"),
    ("LightGBM", "lightgbm", "forecast-tree"),
    ("CatBoost", "catboost", "forecast-tree"),
    ("LSTM", "torch", "forecast-gru"),
    ("TCN", "torch", "forecast-gru"),
)


def advanced_models_enabled() -> bool:
    return os.getenv("MEAL_HISTORY_ENABLE_ADVANCED", "").casefold() in {
        "1", "true", "yes", "on",
    }


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def advanced_forecasts(values: list[float]) -> list[dict[str, object]]:
    """Fit optional tree and neural regressors on a common holdout window."""
    if not advanced_models_enabled() or len(values) < MIN_GRU_OBSERVATIONS:
        return []

    holdout = max(30, min(73, len(values) // 5))
    if len(values) - holdout < 20:
        return []
    train, actual = values[:-holdout], values[-holdout:]
    results = []
    for name, module, extra in ADVANCED_MODEL_SPECS:
        if not _module_available(module):
            continue
        try:
            predicted, technical = _fit_optional_model(name, train, actual)
        except (ImportError, RuntimeError, ValueError):
            continue
        if not predicted:
            continue
        results.append({
            "model": name,
            "actual": actual,
            "predicted": predicted,
            "technical": {
                **technical,
                "training_samples": len(train),
                "validation_samples": len(actual),
                "optional_extra": extra,
            },
        })
    return results


def _lag_features(values: list[float], lookback: int = 14):
    import numpy as np

    x, y = [], []
    for index in range(lookback, len(values)):
        recent = values[index - lookback : index]
        calendar = [index % 7, (index % 7) >= 5]
        x.append(recent + calendar)
        y.append(values[index])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _fit_optional_model(name: str, train: list[float], actual: list[float]):
    import numpy as np

    lookback = min(14, len(train) // 3)
    if name in {"Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"}:
        x_train, y_train = _lag_features(train, lookback)
        if name == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=1)
        elif name == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            estimator = GradientBoostingRegressor(n_estimators=120, random_state=42)
        elif name == "XGBoost":
            from xgboost import XGBRegressor
            estimator = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, n_jobs=1, random_state=42)
        elif name == "LightGBM":
            from lightgbm import LGBMRegressor
            estimator = LGBMRegressor(n_estimators=120, learning_rate=0.05, verbosity=-1, random_state=42)
        else:
            from catboost import CatBoostRegressor
            estimator = CatBoostRegressor(iterations=120, depth=5, learning_rate=0.05, verbose=False, random_seed=42)
        estimator.fit(x_train, y_train)
        history = train[:]
        predicted = []
        for index, _ in enumerate(actual, start=len(train)):
            features = np.asarray(history[-lookback:] + [index % 7, (index % 7) >= 5], dtype=float).reshape(1, -1)
            value = max(0.0, float(estimator.predict(features)[0]))
            predicted.append(value)
            history.append(value)
        return predicted, {"lookback": lookback, "estimator": name}

    import torch
    from torch import nn
    torch.manual_seed(42)
    torch.set_num_threads(1)
    x_train, y_train = _lag_features(train, lookback)
    mean = float(np.mean(train))
    scale = float(np.std(train)) or 1.0
    x_train[:, :lookback] = (x_train[:, :lookback] - mean) / scale
    y_train = (y_train - mean) / scale

    class SequenceNet(nn.Module):
        def __init__(self):
            super().__init__()
            if name == "LSTM":
                self.sequence = nn.LSTM(1, 12, batch_first=True)
            else:
                self.sequence = nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=3, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(8, 8, kernel_size=3, padding=2),
                    nn.ReLU(),
                )
            self.output = nn.Linear(12 if name == "LSTM" else 8, 1)

        def forward(self, batch):
            if name == "LSTM":
                result, _ = self.sequence(batch)
                return self.output(result[:, -1, :])
            result = self.sequence(batch.transpose(1, 2))[:, :, -1]
            return self.output(result)

    model = SequenceNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    x_tensor = torch.tensor(x_train[:, :lookback], dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_function(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    history = train[:]
    predicted = []
    for _ in actual:
        recent = np.asarray([(value - mean) / scale for value in history[-lookback:]], dtype=np.float32)
        batch = torch.tensor(recent).view(1, lookback, 1)
        with torch.no_grad():
            value = max(0.0, float(model(batch).item()) * scale + mean)
        predicted.append(value)
        history.append(value)
    return predicted, {
        "lookback": lookback,
        "hidden_size": 12,
        "epochs": 60,
        "learning_rate": 0.01,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
