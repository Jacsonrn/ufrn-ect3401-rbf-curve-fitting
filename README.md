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

## 📁 Estrutura do Repositório (Planejada)
- 📂 `src/` — Módulos em Python puro e NumPy (`rbf_model.py`, `polynomial_mmq.py`, `metrics.py`).
- 📂 `notebooks/` — Cadernos experimentais e geração de gráficos de superfície e resíduos.
- 📂 `docs/` — Relatório final em LaTeX e slides da apresentação.
- 📄 `Plano Projeto 02.txt` — Cronograma detalhado de execução.

---

## 🚀 Como Executar
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

---

## 👨‍💻 Autor
**Jacson Arruda Ribeiro**  
Universidade Federal do Rio Grande do Norte (UFRN) — Instituto Metrópole Digital (IMD / ECT)
