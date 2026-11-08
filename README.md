# 📉 Non-Linear Curve Fitting: RBF Gaussian Networks vs. Ordinary Least Squares (OLS)

Projeto desenvolvido para a disciplina **ECT-3401 — Computação Numérica** na Universidade Federal do Rio Grande do Norte (UFRN).  
**Docente:** Prof. Joilson B. A. Rego  
**Discente:** Jacson Arruda Ribeiro  

---

## 🎯 Visão Geral
Este repositório contém a implementação analítica e a avaliação comparativa entre **Redes de Base Radial (RBF Gaussianas)** e o método clássico de **Mínimos Quadrados (MMQ / OLS)** para identificação e ajuste de sistemas dinâmicos contínuos.

O foco central é a investigação de baixo nível em Álgebra Linear Numérica, avaliando:
1. **Estabilidade Matricial:** Condicionamento da matriz de projeto $\text{cond}(\Phi^T \Phi)$ sob diferentes aberturas $\sigma$ e número de centros $M$.
2. **Convergência e Calibração:** Minimização do Erro Médio Quadrático (EMQ / RMSE) e regressão linear entre valores reais ($y$) e estimados ($\hat{y}$).
3. **Associação Estatística:** Coeficiente de Correlação de Pearson ($r$) em dados não-vistos (conjunto de teste).

---

## 📁 Estrutura do Repositório
```text
.
├── src/                      # Módulos centrais dos algoritmos
│   ├── data_loader.py        # Divisão treino/teste e MinMax sem data leakage
│   ├── rbf_model.py          # RBF Gaussiana, matriz Phi, cond(Phi^T Phi) e pinv
│   ├── polynomial_mmq.py     # Modelo de Mínimos Quadrados (MMQ / OLS)
│   └── metrics.py            # Cálculo de EMQ, RMSE, Pearson e regressão y vs y_hat
├── tests/                    # Validações experimentais e testes de etapas
│   ├── test_data_loader.py   # Validação de amostragem e isolamento de teste (Dia 1)
│   ├── test_rbf.py           # Validação da RBF e geração de curvas de resposta (Dia 2)
│   ├── test_metrics.py       # Avaliação de EMQ, Pearson e calibração da RBF (Dia 3)
│   └── test_comparison_mmq.py# Comparativo isonômico: RBF vs. MMQ Graus 1, 2 e 3 (Dia 4)
├── docs/                     # Documentação e artefatos de entrega
│   ├── figuras/              # Gráficos de alta resolução para relatório e README
│   └── relatorio/            # Artigo formal em LaTeX e PDF compilado (Dia 5)
├── fundamentacao_teorica.md  # Teoria exaustiva e banco de 66 questões para defesa oral
├── .gitignore                # Regras de exclusão de artefatos temporários
└── README.md                 # Apresentação do projeto e guia de execução
```

---

## 📊 Síntese Comparativa de Desempenho (Isonomia Amostral)

| Modelo | Parâmetros Livres | RMSE Treino ($N=353$) | RMSE Teste ($N=89$) | Pearson $r$ (Teste) | Número de Condição ($\kappa$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RBF Gaussiana ($M=8, \sigma=0{,}1429$)** | 9 | **60,95** | 67,05 | 0,4653 | $1,42 \times 10^5$ |
| **MMQ Grau 1 (Linear)** | 2 | 62,08 | **63,73** | **0,5076** | **36,2** |
| **MMQ Grau 2 (Parábola)**| 3 | 62,04 | 63,91 | 0,5034 | 970 |
| **MMQ Grau 3 (Cúbica)**  | 4 | 62,04 | 63,75 | 0,5052 | 24.300 |

* **Treino:** A RBF Gaussiana superou o MMQ graças à sua flexibilidade local (baixo viés).
* **Teste:** O MMQ Linear superou a RBF diante de dados inéditos, ilustrando o princípio da **Navalha de Occam** (baixa variância e regularização natural contra o ruído biológico).
* **Estabilidade Numérica:** O número de condição de Vandermonde saltou de $36{,}2$ para $24.300$ no Grau 3, comprovando o risco de perda de precisão em ponto flutuante (**IEEE 754**).

---

## 🚀 Como Executar

### 1. Clonar e Configurar o Ambiente
```bash
# Clone o repositório
git clone https://github.com/Jacsonrn/ufrn-ect3401-rbf-curve-fitting.git
cd ufrn-ect3401-rbf-curve-fitting

# Crie e ative o ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instale os pacotes básicos
pip install numpy matplotlib scikit-learn
```

### 2. Executar a Suíte Completa de Validação
```bash
# Validação do Dia 1 (Carregamento e Normalização Min-Max sem Leakage)
python tests/test_data_loader.py

# Validação do Dia 2 (Ajuste da RBF Gaussiana via SVD)
python tests/test_rbf.py

# Validação do Dia 3 (Avaliação Estatística: EMQ, RMSE, Pearson e Calibração OLS)
python tests/test_metrics.py

# Validação do Dia 4 (Comparativo Isonômico: RBF vs. MMQ Graus 1, 2 e 3)
python tests/test_comparison_mmq.py
```

### 3. Visualizar o Relatório Técnico Formal
O artigo científico final estruturado para publicação acadêmica está disponível em:
* Código-fonte LaTeX: [`docs/relatorio/relatorio_final_projeto_u2.tex`](docs/relatorio/relatorio_final_projeto_u2.tex)
* Documento PDF compilado: [`docs/relatorio/relatorio_final_projeto_u2.pdf`](docs/relatorio/relatorio_final_projeto_u2.pdf)

---

## 👨‍💻 Autor
**Jacson Arruda Ribeiro**  
Universidade Federal do Rio Grande do Norte (UFRN) — Instituto Metrópole Digital (IMD / ECT)
