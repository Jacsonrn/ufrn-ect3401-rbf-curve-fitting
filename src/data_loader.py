"""
Módulo de Carregamento e Pré-Processamento de Dados
Projeto U2 — ECT-3401 (Computação Numérica)
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Carrega o dataset load_diabetes, extrai a variável de entrada e o alvo,
    realiza a divisão estrita entre treino e teste, e aplica a normalização MinMax.
    """
    def __init__(self, feature_name: str = "bmi", test_size: float = 0.20, random_state: int = 42):
        self.feature_name = feature_name
        self.test_size = test_size
        self.random_state = random_state
        self.u_min = None
        self.u_max = None

    def load_and_preprocess(self):
        # 1. Carrega o dataset com os nomes originais das features
        data = load_diabetes(as_frame=True)
        df = data.frame

        if self.feature_name not in df.columns:
            raise ValueError(f"Feature '{self.feature_name}' não encontrada. Opções: {list(df.columns)}")

        u_raw = df[self.feature_name].to_numpy()
        y_raw = df["target"].to_numpy()

        # 2. Divisão estrita em Treino (80%) e Teste (20%) com semente fixa para reprodutibilidade
        u_train, u_test, y_train, y_test = train_test_split(
            u_raw, y_raw, test_size=self.test_size, random_state=self.random_state
        )

        # 3. Ajuste do escalonador MinMax APENAS nos dados de treino (evita data leakage)
        self.u_min = float(np.min(u_train))
        self.u_max = float(np.max(u_train))

        # 4. Transformação para a escala [0, 1]
        u_train_norm = (u_train - self.u_min) / (self.u_max - self.u_min)
        u_test_norm = (u_test - self.u_min) / (self.u_max - self.u_min)

        return {
            "u_train": u_train_norm,
            "y_train": y_train,
            "u_test": u_test_norm,
            "y_test": y_test,
            "u_train_raw": u_train,
            "u_test_raw": u_test,
            "scaler_params": {"min": self.u_min, "max": self.u_max},
            "feature_name": self.feature_name
        }
