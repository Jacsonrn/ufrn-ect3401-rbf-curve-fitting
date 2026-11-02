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
│   ├── test_data_loader.py   # Validação de amostragem e isolamento de teste
│   └── test_rbf.py           # Validação da RBF e geração de curvas de resposta
├── docs/                     # Documentação e artefatos de entrega
│   ├── figuras/              # Gráficos de alta resolução para relatório e README
│   └── relatorio/            # Artigo/Relatório formal em LaTeX
├── fundamentacao_teorica.md  # Teoria exaustiva e banco de questões para defesa oral
├── Plano Projeto 02.txt      # Cronograma detalhado de execução (5 Dias)
├── .gitignore                # Regras de exclusão de artefatos temporários
└── README.md                 # Apresentação do projeto e guia de execução
```

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

### 2. Executar os Testes de Validação
```bash
# Validação do Dia 1 (Carregamento e Normalização)
python tests/test_data_loader.py

# Validação do Dia 2 (Ajuste da RBF Gaussiana)
python tests/test_rbf.py
```

---

## 👨‍💻 Autor
**Jacson Arruda Ribeiro**  
Universidade Federal do Rio Grande do Norte (UFRN) — Instituto Metrópole Digital (IMD / ECT)
