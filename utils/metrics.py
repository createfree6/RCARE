from __future__ import annotations

import numpy as np


def RSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2) + 1e-8))


def CORR(pred: np.ndarray, true: np.ndarray) -> float:
    pred_center = pred - pred.mean(axis=0, keepdims=True)
    true_center = true - true.mean(axis=0, keepdims=True)
    corr = (pred_center * true_center).sum(axis=0) / (
        np.sqrt((pred_center**2).sum(axis=0) * (true_center**2).sum(axis=0)) + 1e-8
    )
    return float(np.nanmean(corr))


def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs((pred - true) / (true + 1e-8))))


def MSPE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.square((pred - true) / (true + 1e-8))))


def metric(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float, float, float]:
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe
