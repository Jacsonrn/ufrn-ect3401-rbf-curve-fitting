"""
Script de Validação do Dia 1: Carregamento e Espalhamento Treino vs Teste
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import DataLoader

def main():
    loader = DataLoader(feature_name="bmi", test_size=0.20, random_state=42)
    data = loader.load_and_preprocess()

    print(f"Dataset carregado com sucesso:")
    print(f" - Amostras de Treino : {len(data['u_train'])} (~80%)")
    print(f" - Amostras de Teste  : {len(data['u_test'])} (~20%)")
    print(f" - Intervalo Treino u : [{data['u_train'].min():.4f}, {data['u_train'].max():.4f}]")
    print(f" - Intervalo Teste u  : [{data['u_test'].min():.4f}, {data['u_test'].max():.4f}]")

    # Plot do espalhamento
    plt.figure(figsize=(10, 5))
    plt.scatter(data['u_train'], data['y_train'], color='navy', alpha=0.6, edgecolors='k', s=35, label='Treino (80%)')
    plt.scatter(data['u_test'], data['y_test'], color='crimson', alpha=0.8, marker='s', s=40, label='Teste (20%)')
    plt.title("Dia 1: Espalhamento dos Dados Normalizados (BMI vs Progressão da Diabetes)", fontsize=12)
    plt.xlabel("Entrada Normalizada u (BMI) [0, 1]", fontsize=11)
    plt.ylabel("Alvo y (Progressão da Doença)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dia1_espalhamento_dados.png", dpi=300)
    print("Gráfico salvo como 'dia1_espalhamento_dados.png'.")

if __name__ == "__main__":
    main()
