# Por que uma simples reta superou uma Rede RBF não-linear? O dilema de Occam e o IEEE 754 na prática

Muitos desenvolvedores e cientistas de dados acreditam que um modelo não-linear com maior número de parâmetros sempre vencerá uma regressão linear clássica. 

Na prática da Computação Numérica, a realidade é implacável: modelos mais complexos frequentemente caem em armadilhas de sobreajuste e instabilidade matricial quando expostos a ruídos biológicos reais.

Recentemente, finalizei a implementação de um projeto na UFRN confrontando **Redes de Base Radial (RBF Gaussiana)** com o **Método dos Mínimos Quadrados (MMQ Polinomial)** para estimar a progressão clínica do diabetes a partir do IMC de 442 pacientes (`load_diabetes`).

Aqui está o diagnóstico de baixo nível e as lições aprendidas:

---

### 🎯 1. Gancho do Problema Real (Situation & Task)
Em dados biomédicos, as correlações raramente são puras: existe dispersão estocástica intrínseca e medições com ruído. Se o pipeline computacional permitir **vazamento de dados (*data leakage*)** — como calcular parâmetros de normalização Min-Max no conjunto total antes do split —, as métricas de validação tornam-se falsas e perigosas para tomadas de decisão clínica. O desafio era construir um pipeline 100% isonômico (80% treino / 20% teste em quarentena estrita) para comparar o ajuste de curvas local vs. global.

---

### ⚙️ 2. Diagnóstico Low-Level de SO & Álgebra Numérica (Troubleshooting)
Ao elevar a ordem do modelo MMQ na base monomial (Matriz de Vandermonde $V$), observamos uma explosão no **Número de Condição**:
* **MMQ Grau 1 (Linear):** $\kappa = 36{,}2$
* **MMQ Grau 2 (Parábola):** $\kappa = 970$
* **MMQ Grau 3 (Cúbica):** $\kappa = 24.300$

No padrão **IEEE 754** (ponto flutuante de 64 bits com $\approx 16$ dígitos de precisão), um número de condição de ordem $10^4$ significa a perda imediata de **4 a 5 dígitos decimais por erros de cancelamento catastrófico** ao calcular a inversa $(\Phi^T \Phi)^{-1}$.

Da mesma forma, na RBF com 8 centros gaussianos uniformes ($\sigma = 0{,}1429$), a proximidade dos centros gerou $\kappa = 1{,}42 \times 10^5$. Se tentássemos a inversão direta via `np.linalg.inv`, o sistema entraria em colapso numérico. A saída mandatória foi recorrer à **Decomposição em Valores Singulares (SVD)** através da pseudoinversa de Moore-Penrose (`np.linalg.pinv`), truncando autovalores virtualmente nulos e garantindo coeficientes estáveis.

---

### 🏗️ 3. Arquitetura e Solução Adotada (Action)
* **Prevenção estrita de Data Leakage:** O escalonamento Min-Max $[0, 1]$ foi parametrizado exclusivamente sobre os $N=353$ pacientes de treino e congelado para projetar os $N=89$ pacientes de teste.
* **Modelo RBF Gaussiana:** $M=8$ centros com bases radiais de suporte local $\phi_j(u) = \exp\left(-\frac{\|u-c_j\|^2}{2\sigma^2}\right)$ e termo de bias $w_0$.
* **MMQ Polinomial:** Montagem explícita da Matriz de Vandermonde para ordens 1, 2 e 3 resolvidas via Equação Normal.
* **Métricas Isonômicas:** Avaliação comparativa de $\text{EMQ}$, $\text{RMSE}$, Coeficiente de Correlação de Pearson ($r$) e reta de calibração $y = a\hat{y} + b$.

---

### 📊 4. Impacto Prático Mensurável (Result)
Os resultados no conjunto de teste independente revelaram uma aula prática de Ciência de Dados:
1. **No Treino (Capacidade de Ajuste / Baixo Viés):** A RBF Gaussiana superou todos os concorrentes ($\text{RMSE} = 60{,}95$ vs. $62{,}08$ do MMQ Linear), graças aos seus 9 graus de liberdade locais que se adaptaram aos vales dos dados.
2. **No Teste (Generalização / Baixa Variância):** O **MMQ Linear venceu** com folga ($\text{RMSE} = 63{,}73$ contra $67{,}05$ da RBF) e maior correlação linear de Pearson ($r = 0{,}5076$ contra $0{,}4653$).
3. **A Navalha de Occam na prática:** Modelos quadráticos e cúbicos aumentaram o custo e o mau condicionamento sem reduzir o erro de teste ($63{,}91$ e $63{,}75$). Diante de forte ruído biológico, a simplicidade de 2 parâmetros atuou como um regularizador natural, recusando-se a memorizar ruído.

**Conclusão de Engenharia (*No Free Lunch Theorem*):** A RBF é insubstituível para fenômenos físicos com dinâmicas locais ricas (saturação de máquinas, transientes), mas o MMQ Linear continua imbatível em estabilidade e parcimônia para relações estocásticas ruidosas.

---

### 🔗 Código Aberto e Relatório Acadêmico:
O projeto completo com suíte de testes unitários automatizados, gráficos de resíduos e o relatório formal em LaTeX compilado em PDF está disponível no GitHub:
👉 https://github.com/Jacsonrn/ufrn-ect3401-rbf-curve-fitting

#ComputacaoNumerica #MachineLearning #DataScience #LinearAlgebra #Python #IEEE754 #DataLeakage #SVD #OccamsRazor #EngenhariaDeSoftware #UFRN
