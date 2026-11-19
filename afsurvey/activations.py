"""NumPy definitions of activation functions f(x) and their derivatives f'(x)."""
import numpy as np
from scipy import special

SELU_LAMBDA = 1.0507009873554805
SELU_ALPHA = 1.6732632423543772


def sigmoid(x):
    return special.expit(x)


def softplus(x):
    return np.logaddexp(0.0, x)


def _neg(x):
    return np.minimum(x, 0.0)


ACTIVATIONS = {
    "Sigmoid": (sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
    "Tanh": (np.tanh, lambda x: 1 - np.tanh(x) ** 2),
    "ReLU": (lambda x: np.maximum(0.0, x), lambda x: (x > 0).astype(float)),
    "Leaky ReLU": (lambda x: np.where(x > 0, x, 0.01 * x), lambda x: np.where(x > 0, 1.0, 0.01)),
    "ELU": (lambda x: np.where(x > 0, x, np.expm1(_neg(x))),
            lambda x: np.where(x > 0, 1.0, np.exp(_neg(x)))),
    "SELU": (lambda x: SELU_LAMBDA * np.where(x > 0, x, SELU_ALPHA * np.expm1(_neg(x))),
             lambda x: SELU_LAMBDA * np.where(x > 0, 1.0, SELU_ALPHA * np.exp(_neg(x)))),
    "Swish": (lambda x: x * sigmoid(x),
              lambda x: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))),
    "GELU": (lambda x: x * 0.5 * (1 + special.erf(x / np.sqrt(2))),
             lambda x: 0.5 * (1 + special.erf(x / np.sqrt(2))) + x * np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)),
    "Mish": (lambda x: x * np.tanh(softplus(x)),
             lambda x: np.tanh(softplus(x)) + x * sigmoid(x) * (1 - np.tanh(softplus(x)) ** 2)),
    "LiSHT": (lambda x: x * np.tanh(x), lambda x: np.tanh(x) + x * (1 - np.tanh(x) ** 2)),
    # Rectified Tangent Activation: identity for x >= 0, tanh for x < 0
    "RTA": (lambda x: np.maximum(x, np.tanh(x)),
            lambda x: np.where(x > 0, 1.0, 1 - np.tanh(x) ** 2)),
}


def get(name):
    """Return (f, f_prime) for an activation name."""
    return ACTIVATIONS[name]
