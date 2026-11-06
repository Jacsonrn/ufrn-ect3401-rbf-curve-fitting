"""
Script de Validação do Dia 4: Comparativo RBF vs. MMQ (Graus 1, 2 e 3)
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
from src.polynomial_mmq import PolynomialMMQRegressor
from src.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    pearson_correlation,
    linear_regression_fit
)

def main():
    loader = DataLoader(feature_name="bmi", test_size=0.20, random_state=42)
    data = loader.load_and_preprocess()

    models = {
        "RBF (M=8, σ=0.14)": RBFGaussianRegressor(n_centers=8),
        "MMQ Grau 1 (Linear)": PolynomialMMQRegressor(degree=1),
        "MMQ Grau 2 (Parábola)": PolynomialMMQRegressor(degree=2),
        "MMQ Grau 3 (Cúbico)": PolynomialMMQRegressor(degree=3),
    }

    results = {}
    print("=" * 80)
    print("  TABELA COMPARATIVA OFICIAL: RBF vs. MMQ (DIA 4)")
    print("=" * 80)
    print(f"{'Modelo':<23} | {'RMSE Treino':<11} | {'RMSE Teste':<10} | {'Pearson r (Teste)':<17} | {'cond(Phi^T Phi)':<14}")
    print("-" * 80)

    for name, model in models.items():
        model.fit(data['u_train'], data['y_train'])
        
        y_tr_pred = model.predict(data['u_train'])
        y_te_pred = model.predict(data['u_test'])
        
        rmse_tr = root_mean_squared_error(data['y_train'], y_tr_pred)
        rmse_te = root_mean_squared_error(data['y_test'], y_te_pred)
        r_te = pearson_correlation(data['y_test'], y_te_pred)
        reg_te = linear_regression_fit(data['y_test'], y_te_pred)
        
        results[name] = {
            "model": model,
            "rmse_train": rmse_tr,
            "rmse_test": rmse_te,
            "pearson_r": r_te,
            "cond_number": model.cond_number_,
            "y_test_pred": y_te_pred
        }
        
        print(f"{name:<23} | {rmse_tr:<11.2f} | {rmse_te:<10.2f} | {r_te:<17.4f} | {model.cond_number_:<14.2e}")
    print("=" * 80)

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "figuras")
    u_grid = np.linspace(0.0, 1.0, 300)

    # GRÁFICO 1: Curvas Sobrepostas
    plt.figure(figsize=(11, 6))
    plt.scatter(data['u_train'], data['y_train'], color='gray', alpha=0.25, s=20, label='Treino (80%)')
    plt.scatter(data['u_test'], data['y_test'], color='crimson', alpha=0.7, marker='s', s=35, label='Teste (20%)')
    
    colors = {
        "RBF (M=8, σ=0.14)": ('forestgreen', 2.8, '-'),
        "MMQ Grau 1 (Linear)": ('navy', 1.8, '--'),
        "MMQ Grau 2 (Parábola)": ('darkorange', 2.0, '-.'),
        "MMQ Grau 3 (Cúbico)": ('purple', 1.8, ':')
    }
    
    for name, info in results.items():
        c, lw, ls = colors[name]
        y_grid_pred = info["model"].predict(u_grid)
        plt.plot(u_grid, y_grid_pred, color=c, linewidth=lw, linestyle=ls,
                 label=f'{name} — RMSE: {info["rmse_test"]:.2f} | r: {info["pearson_r"]:.3f}')
        
    plt.title("Dia 4: Comparação de Modelos — RBF Gaussiana vs. Mínimos Quadrados (MMQ)", fontsize=12)
    plt.xlabel("Entrada Normalizada u (BMI)", fontsize=11)
    plt.ylabel("Alvo y (Progressão da Doença)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "dia4_comparativo_rbf_vs_mmq.png")
    plt.savefig(p1, dpi=300)
    plt.close()

    # GRÁFICO 2: Barras Comparativas de Desempenho no Teste
    nomes = list(results.keys())
    rmses = [results[m]["rmse_test"] for m in nomes]
    pearsons = [results[m]["pearson_r"] for m in nomes]
    
    x = np.arange(len(nomes))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    color = 'tab:red'
    ax1.set_ylabel('RMSE no Teste (Menor é Melhor)', color=color, fontsize=11)
    bars1 = ax1.bar(x - width/2, rmses, width, label='RMSE Teste', color='coral', edgecolor='k')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(50, 75)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Pearson r no Teste (Maior é Melhor)', color=color, fontsize=11)
    bars2 = ax2.bar(x + width/2, pearsons, width, label='Pearson r', color='royalblue', edgecolor='k')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.40, 0.50)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes, rotation=15, ha='right', fontsize=9)
    plt.title("Dia 4: Avaliação Isonômica no Conjunto de Teste (RMSE vs. Pearson r)", fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(fig_dir, "dia4_barras_comparativas_rmse.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    print(f"Gráficos salvos com sucesso em '{fig_dir}'.")

if __name__ == "__main__":
    main()
