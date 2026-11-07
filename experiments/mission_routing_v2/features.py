"""Continuous, inexpensive public RGB features; no filename, clock or target IDs."""

from __future__ import annotations

import numpy as np

NAMES = (
    "bias",
    "luminance",
    "contrast",
    "dark_fraction",
    "bright_fraction",
    "horizontal_gradient",
    "vertical_gradient",
    "color_spread",
    "lower_luminance",
    "upper_luminance",
    "contrast_squared",
    "gradient_product",
)


def encode(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected native uint8 RGB")
    # Fixed grid bounds feature acquisition work independently of sensor resolution.
    image = rgb[:: max(1, rgb.shape[0] // 48), :: max(1, rgb.shape[1] // 64)].astype(float) / 255
    gray = image @ np.array([0.2126, 0.7152, 0.0722])
    gx, gy = np.abs(np.diff(gray, axis=1)).mean(), np.abs(np.diff(gray, axis=0)).mean()
    contrast = gray.std()
    result = np.array(
        [
            1,
            gray.mean(),
            contrast,
            (gray < 0.2).mean(),
            (gray > 0.8).mean(),
            gx,
            gy,
            (image.max(axis=2) - image.min(axis=2)).mean(),
            gray[len(gray) // 2 :].mean(),
            gray[: len(gray) // 2].mean(),
            contrast**2,
            gx * gy,
        ]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite RGB features")
    return result


def fit(x, y, alpha=1.0):
    """Full-information offline ridge, not an online reward oracle or RL reproduction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or x.shape[1] != len(NAMES) or len(x) != len(y) or not len(x):
        raise ValueError("Complete training feature/label pairs required")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    mean[0], scale[0] = 0, 1
    scale = np.maximum(scale, 0.01)
    z = (x - mean) / scale
    inverse = np.linalg.inv(z.T @ z + alpha * np.eye(z.shape[1]))
    coefficient = inverse @ z.T @ y
    return dict(
        mean=mean.tolist(),
        scale=scale.tolist(),
        coefficient=coefficient.tolist(),
        inverse=inverse.tolist(),
        alpha=alpha,
        samples=len(x),
        global_mean=float(y.mean()),
    )


def predict(model, x):
    z = (np.asarray(x) - model["mean"]) / model["scale"]
    value = float(np.clip(z @ model["coefficient"], 0, 1))
    uncertainty = float(np.sqrt(max(0, z @ np.asarray(model["inverse"]) @ z)))
    return value, uncertainty
