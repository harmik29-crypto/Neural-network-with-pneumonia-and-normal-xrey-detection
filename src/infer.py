"""Loading saved models and running inference in memory-friendly chunks."""
import json

import keras
import numpy as np

from .config import MODELS_DIR
from .preprocess import to_model_input
from scipy.optimize import minimize_scalar
from scipy.special import expit

def load_model(name: str) -> keras.Model:
    return keras.saving.load_model(MODELS_DIR / f"{name}.keras", compile=False)


def load_meta(name: str) -> dict:
    return json.load(open(MODELS_DIR / f"{name}_meta.json"))


def load_threshold(name: str) -> float:
    return json.load(open(MODELS_DIR / f"{name}_threshold.json"))["threshold"]

def predict_logit(model, x_u8, chunk=128):
    out = [model.predict(to_model_input(x_u8[i:i + chunk]), batch_size=64, verbose=0).reshape(-1)
           for i in range(0, len(x_u8), chunk)]
    return np.concatenate(out)

def predict_prob(model, x_u8, chunk=128, T=1.0):
    return expit(predict_logit(model, x_u8, chunk) / T)

def fit_temperature(logits, y):
    def nll(T):
        p = np.clip(expit(logits / T), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.5, 20), method="bounded").x)

def load_temperature(name):
    return load_meta(name).get("temperature", 1.0)