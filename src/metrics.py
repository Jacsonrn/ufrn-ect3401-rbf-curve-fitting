"""
Módulo de Métricas Estatísticas e Avaliação de Modelos
Projeto U2 — ECT-3401 (Computação Numérica)
"""
import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Erro Médio Quadrático (EMQ / MSE)."""
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(errors ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a Raiz do Erro Médio Quadrático (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Coeficiente de Correlação Linear de Pearson (r).
    r = cov(y_true, y_pred) / (std(y_true) * std(y_pred))
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    diff_t = y_t - np.mean(y_t)
    diff_p = y_p - np.mean(y_p)
    
    numerator = np.sum(diff_t * diff_p)
    denominator = np.sqrt(np.sum(diff_t ** 2) * np.sum(diff_p ** 2))
    
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def linear_regression_fit(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Ajusta a reta de calibração y_true = a * y_pred + b via Mínimos Quadrados.
    Ideal: a = 1.0, b = 0.0
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    # Matriz [y_pred, 1] para resolver [a, b]^T
    A = np.vstack([y_p, np.ones(len(y_p))]).T
    a, b = np.linalg.lstsq(A, y_t, rcond=None)[0]
    
    r = pearson_correlation(y_t, y_p)
    return {
        "slope": float(a),
        "intercept": float(b),
        "pearson_r": float(r),
        "r2_score": float(r ** 2)
    }
