"""
Módulo do Modelo de Regressão por Funções de Base Radial (RBF Gaussiana)
Projeto U2 — ECT-3401 (Computação Numérica)
"""
import numpy as np


class RBFGaussianRegressor:
    """
    Regressor RBF baseado em Funções Gaussianas com resolução analítica via Equação Normal.
    """
    def __init__(self, n_centers: int = 8, sigma: float = None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers_ = None
        self.sigma_ = None
        self.weights_ = None
        self.cond_number_ = None

    def _init_centers_and_sigma(self, u: np.ndarray):
        # Distribui os M centros uniformemente ao longo do domínio normalizado [0, 1]
        self.centers_ = np.linspace(0.0, 1.0, self.n_centers)
        
        # Heurística analítica: sigma proporcional ao espaçamento entre centros adjacentes
        spacing = self.centers_[1] - self.centers_[0]
        if self.sigma is None:
            self.sigma_ = float(spacing)
        else:
            self.sigma_ = float(self.sigma)

    def _compute_phi(self, u: np.ndarray) -> np.ndarray:
        """Calcula a Matriz de Projeto Phi com termo de viés (primeira coluna = 1)."""
        u = np.asarray(u).reshape(-1, 1)  # Forma (N, 1)
        c = self.centers_.reshape(1, -1)   # Forma (1, M)
        
        # Distância quadrática via broadcasting NumPy em tempo O(N * M)
        sq_dist = (u - c) ** 2
        phi_gaussian = np.exp(-sq_dist / (2.0 * (self.sigma_ ** 2)))
        
        # Adiciona a coluna de viés w0 na 1ª coluna (N, M + 1)
        bias = np.ones((len(u), 1))
        return np.hstack([bias, phi_gaussian])

    def fit(self, u: np.ndarray, y: np.ndarray):
        """Ajusta os pesos w resolvendo o sistema linear da Equação Normal."""
        self._init_centers_and_sigma(u)
        phi = self._compute_phi(u)
        
        # Avaliação de estabilidade numérica via Número de Condição
        phi_t_phi = phi.T @ phi
        self.cond_number_ = float(np.linalg.cond(phi_t_phi))
        
        # Resolução estável via pseudo-inversa de Moore-Penrose (SVD)
        self.weights_ = np.linalg.pinv(phi) @ y
        return self

    def predict(self, u: np.ndarray) -> np.ndarray:
        """Gera a predição y_hat = Phi @ w."""
        if self.weights_ is None:
            raise RuntimeError("O modelo precisa ser ajustado (fit) antes de realizar predições.")
        phi = self._compute_phi(u)
        return phi @ self.weights_
