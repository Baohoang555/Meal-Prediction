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
