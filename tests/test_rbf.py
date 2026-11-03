"""
Script de Validação do Dia 2: Ajuste da RBF Gaussiana e Curva de Resposta
"""
import os
import sys
# Garante saída UTF-8 para evitar falhas do cp1252 com caracteres gregos
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import DataLoader
from src.rbf_model import RBFGaussianRegressor

def main():
    loader = DataLoader(feature_name="bmi", test_size=0.20, random_state=42)
    data = loader.load_and_preprocess()

    # Instancia e ajusta a RBF com M=8 centros
    rbf = RBFGaussianRegressor(n_centers=8)
    rbf.fit(data['u_train'], data['y_train'])

    print("=" * 60)
    print("  RELATÓRIO DO AJUSTE RBF GAUSSIANA (DIA 2)")
    print("=" * 60)
    print(f" - Quantidade de Centros (M) : {rbf.n_centers}")
    print(f" - Abertura Gaussiana (sigma): {rbf.sigma_:.4f}")
    print(f" - Número de Condição        : {rbf.cond_number_:.2e}")
    print(f" - Termo de Viés (w0)        : {rbf.weights_[0]:.2f}")
    print(f" - Pesos das Bases (w1..wM)  : {np.round(rbf.weights_[1:], 2)}")
    print("=" * 60)

    # Geração de grade densa para traçar a curva contínua suave
    u_grid = np.linspace(0.0, 1.0, 300)
    y_grid_pred = rbf.predict(u_grid)

    # Plot do ajuste sobre os dados
    plt.figure(figsize=(10, 5))
    plt.scatter(data['u_train'], data['y_train'], color='navy', alpha=0.4, s=30, label='Treino (80%)')
    plt.scatter(data['u_test'], data['y_test'], color='crimson', alpha=0.7, marker='s', s=35, label='Teste (20%)')
    plt.plot(u_grid, y_grid_pred, color='forestgreen', linewidth=2.5, label=f'RBF Ajustada (M={rbf.n_centers}, sigma={rbf.sigma_:.3f})')
    plt.scatter(rbf.centers_, np.full_like(rbf.centers_, data['y_train'].min() - 10), 
                color='black', marker='^', s=60, label=f'Centros $c_j$ ($M={rbf.n_centers}$)')
    
    plt.title(rf"Dia 2: Ajuste via RBF Gaussiana ($M={rbf.n_centers}, \sigma={rbf.sigma_:.3f}$) - cond($\Phi^T\Phi$)={rbf.cond_number_:.1e}", fontsize=11)
    plt.xlabel("Entrada Normalizada u (BMI)", fontsize=11)
    plt.ylabel("Alvo y (Progressão da Doença)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "docs", "figuras", "dia2_ajuste_rbf.png")
    plt.savefig(output_path, dpi=300)
    print(f"Gráfico salvo com sucesso em '{output_path}'.")

if __name__ == "__main__":
    main()
