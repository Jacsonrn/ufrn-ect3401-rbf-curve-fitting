"""
Script de Validação do Dia 3: EMQ, RMSE, Pearson e Regressão Linear Real vs Predito
"""
import os
import sys
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
from src.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    pearson_correlation,
    linear_regression_fit
)

def main():
    loader = DataLoader(feature_name="bmi", test_size=0.20, random_state=42)
    data = loader.load_and_preprocess()

    rbf = RBFGaussianRegressor(n_centers=8)
    rbf.fit(data['u_train'], data['y_train'])

    y_train_pred = rbf.predict(data['u_train'])
    y_test_pred = rbf.predict(data['u_test'])

    # 1. EMQ e RMSE
    emq_train = mean_squared_error(data['y_train'], y_train_pred)
    rmse_train = root_mean_squared_error(data['y_train'], y_train_pred)
    emq_test = mean_squared_error(data['y_test'], y_test_pred)
    rmse_test = root_mean_squared_error(data['y_test'], y_test_pred)

    # 2. Correlação de Pearson
    r_train = pearson_correlation(data['y_train'], y_train_pred)
    r_test = pearson_correlation(data['y_test'], y_test_pred)

    # 3. Regressão Linear Real vs Estimado no Teste
    reg = linear_regression_fit(data['y_test'], y_test_pred)

    print("=" * 65)
    print("  RELATÓRIO DE AVALIAÇÃO ESTATÍSTICA — RBF (DIA 3)")
    print("=" * 65)
    print(f" Treino -> EMQ: {emq_train:.2f} | RMSE: {rmse_train:.2f} | Pearson r: {r_train:.4f}")
    print(f" Teste  -> EMQ: {emq_test:.2f} | RMSE: {rmse_test:.2f} | Pearson r: {r_test:.4f}")
    print("-" * 65)
    print(" Regressão Linear no Teste (y_real = a * y_pred + b):")
    print(f"  - Coeficiente Angular (a) : {reg['slope']:.4f} (Ideal: 1.0000)")
    print(f"  - Intercepto Linear (b)   : {reg['intercept']:.2f} (Ideal: 0.00)")
    print(f"  - Coef. Determinação (R²) : {reg['r2_score']:.4f}")
    print("=" * 65)

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "figuras")

    # GRÁFICO 1: Curva RBF com Faixa de Incerteza (+- RMSE)
    u_grid = np.linspace(0.0, 1.0, 300)
    y_grid = rbf.predict(u_grid)

    plt.figure(figsize=(10, 5))
    plt.scatter(data['u_train'], data['y_train'], color='navy', alpha=0.3, s=25, label='Treino')
    plt.scatter(data['u_test'], data['y_test'], color='crimson', alpha=0.7, marker='s', s=35, label='Teste')
    plt.plot(u_grid, y_grid, color='forestgreen', linewidth=2.5, label='RBF Estimada')
    plt.fill_between(u_grid, y_grid - rmse_test, y_grid + rmse_test, color='forestgreen', alpha=0.15,
                     label=rf'Faixa de Incerteza ($\pm$RMSE = $\pm${rmse_test:.1f})')
    plt.title(f"Dia 3: Curva RBF com Faixa de Incerteza — RMSE Teste: {rmse_test:.2f}", fontsize=11)
    plt.xlabel("Entrada Normalizada u (BMI)", fontsize=11)
    plt.ylabel("Alvo y (Progressão da Doença)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "dia3_rbf_faixa_incerteza.png")
    plt.savefig(p1, dpi=300)
    plt.close()

    # GRÁFICO 2: Regressão Linear Real vs Estimado
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_pred, data['y_test'], color='crimson', alpha=0.75, edgecolors='k', s=45, label='Amostras de Teste')
    
    y_line = np.linspace(min(y_test_pred), max(y_test_pred), 100)
    plt.plot(y_line, reg['slope'] * y_line + reg['intercept'], color='navy', linewidth=2,
             label=f'Reta Ajustada: $y = {reg["slope"]:.2f}\\hat{{y}} + {reg["intercept"]:.1f}$')
    
    lims = [min(min(y_test_pred), min(data['y_test'])) - 10, max(max(y_test_pred), max(data['y_test'])) + 10]
    plt.plot(lims, lims, color='gray', linestyle='--', linewidth=1.5, label='Ideal ($y = \\hat{y}$)')
    
    plt.title(f"Dia 3: Regressão Real vs Predito (Pearson $r = {reg['pearson_r']:.3f}$)", fontsize=11)
    plt.xlabel(r"Valor Estimado $\hat{y}$ (RBF)", fontsize=11)
    plt.ylabel("Valor Real $y$ (Diabetes)", fontsize=11)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "dia3_regressao_real_vs_predito.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    print(f"Gráficos salvos com sucesso em '{fig_dir}'.")

if __name__ == "__main__":
    main()
