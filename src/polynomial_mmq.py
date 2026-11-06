"""
Módulo do Modelo de Regressão por Mínimos Quadrados (MMQ Polinomial)
Projeto U2 — ECT-3401 (Computação Numérica)
"""
import numpy as np


class PolynomialMMQRegressor:
    """
    Regressor de Mínimos Quadrados Ordinários (MMQ / OLS) Polinomial.
    Suporta Graus 1 (Reta Linear), 2 (Parábola), 3 (Cúbico), etc.
    """
    def __init__(self, degree: int = 1):
        if degree < 1:
            raise ValueError("O grau polinomial deve ser maior ou igual a 1.")
        self.degree = degree
        self.weights_ = None
        self.cond_number_ = None

    def _compute_vandermonde(self, u: np.ndarray) -> np.ndarray:
        """Monta a matriz de Vandermonde [1, u, u^2, ..., u^d]."""
        u = np.asarray(u, dtype=float).ravel()
        # Matriz de potências sucessivas de 0 até degree
        return np.vander(u, N=self.degree + 1, increasing=True)

    def fit(self, u: np.ndarray, y: np.ndarray):
        """Ajusta os coeficientes polinomiais via Equação Normal."""
        V = self._compute_vandermonde(u)
        
        # Avalia o número de condição da matriz Vandermonde
        v_t_v = V.T @ V
        self.cond_number_ = float(np.linalg.cond(v_t_v))
        
        # Solução analítica de Mínimos Quadrados
        self.weights_ = np.linalg.pinv(V) @ np.asarray(y, dtype=float)
        return self

    def predict(self, u: np.ndarray) -> np.ndarray:
        """Gera a predição y_hat = V @ w."""
        if self.weights_ is None:
            raise RuntimeError("O modelo precisa ser ajustado (fit) antes de realizar predições.")
        V = self._compute_vandermonde(u)
        return V @ self.weights_
