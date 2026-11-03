# 🧠 Fundamentação Teórica do Projeto: Ajuste de Curvas via RBF e MMQ

Este documento centraliza a fundamentação matemática, algébrica e computacional por trás de todas as decisões de projeto da disciplina **ECT-3401 — Computação Numérica (UFRN)**, ministrada pelo **Prof. Joilson B. A. Rego**.

O conteúdo está organizado estritamente pelo cronograma de dias de execução, servindo como base técnica exaustiva para consulta, elaboração do relatório final em LaTeX e preparação para arguições e defesas orais.

---

# 📅 DIA 1: Seleção de Dados, Amostragem e Normalização sem Vazamento

Neste primeiro dia, estabelecemos as fundações estatísticas do projeto: a escolha da variável contínua, o protocolo de particionamento estrito entre treino e teste e a transformação geométrica de escala necessária para a estabilidade da base radial.

---

### 📚 1. Contextualização: Identificação de Sistemas e Ajuste de Curvas
Na engenharia e na computação científica, o problema de **ajuste de curvas** (*curve fitting*) consiste em encontrar uma função contínua $f: \mathbb{R} \to \mathbb{R}$ parametrizada por um vetor de pesos $\mathbf{w}$ tal que:
$$y_i = f(u_i; \mathbf{w}) + \epsilon_i, \quad i = 1, \dots, N$$
onde $u_i$ representa a variável de excitação/entrada, $y_i$ é a resposta observada do sistema físico e $\epsilon_i \sim \mathcal{N}(0, \sigma_\epsilon^2)$ representa o ruído estocástico de medição ou variabilidade intrínseca não-modelada.

Diferente de problemas puramente interpolativos (como Splines cúbicas ou polinômios de Lagrange, que forçam a curva a passar exatamente por cada ponto, amplificando o ruído), o objetivo da Unidade 2 é a **filtragem e aproximação suave**: o modelo deve aprender o padrão subjacente (*sinal*) e ignorar as flutuações de alta frequência (*ruído*).

---

### 📚 2. Caracterização do Dataset `load_diabetes`
Para cumprir as diretrizes do [`Projeto_U2.pdf`](file:///C:/Users/Jacson/Documents/Repositorios/Computacao%20Numerica/Unidade%202/Projeto_U2.pdf) — modelagem contínua com dinâmica não-linear e ruído —, selecionamos o dataset clássico de **Efron, Hastie, Johnstone e Tibshirani (2004)**, distribuído nativamente pelo `scikit-learn`:

* **População:** 442 pacientes diabéticos monitorados clinicamente.
* **Variável Alvo ($y$):** Medida quantitativa contínua da progressão da diabetes registrada um ano após as medições de base (valores variando no intervalo real $[25, 346]$).
* **Variável de Entrada Escolhida ($u = \text{BMI}$):** O Índice de Massa Corporal (IMC / *Body Mass Index*).
  * **Justificativa Biofísica:** O IMC apresenta a mais alta correlação bivariada com a resistência insulínica e o agravamento do quadro diabético.
  * **Justificativa Numérica:** A relação entre IMC e progressão clínica não é estritamente linear: para faixas intermediárias de IMC há grande dispersão, e nos extremos ocorrem inflexões fisiológicas. Trata-se do cenário ideal para confrontar a rigidez dos Mínimos Quadrados (MMQ) contra a flexibilidade local das RBFs Gaussianas.

---

### 📚 3. Particionamento Treino vs. Teste e Teoria da Generalização
Avaliamos a capacidade preditiva de qualquer modelo por sua habilidade de **generalizar** — isto é, operar sobre dados não utilizados durante o processo de ajuste dos parâmetros:

* **Conjunto de Treinamento ($N_{\text{treino}} = 353$, ~80%):** Utilizado exclusivamente para construir a matriz de projeto $\Phi$, resolver a Equação Normal e determinar o vetor de pesos $\mathbf{w}$.
* **Conjunto de Teste ($N_{\text{teste}} = 89$, ~20%):** Dados "inéditos" mantidos em cofre para a aferição do Erro Médio Quadrático ($\text{EMQ}$) e do Coeficiente de Correlação de Pearson ($r$).
* **Reprodutibilidade Científica (`random_state=42`):** A função de partição pseudo-aleatória é ancorada em uma semente fixa. Isso assegura que qualquer execução em qualquer máquina gere exatamente a mesma divisão amostral, viabilizando a auditoria e comparação isonômica entre modelos.

---

### 📚 4. Normalização MinMax e a Eliminação do Vazamento de Dados (*Data Leakage*)
O escalonamento por intervalo MinMax é definido algebricamente como:
$$u_{\text{norm}} = \frac{u - u_{\min}}{u_{\max} - u_{\min}}$$

#### O Perigo do Vazamento de Dados (*Data Leakage*):
Se aplicássemos o `MinMaxScaler` sobre a totalidade dos 442 dados antes da divisão:
1. Os valores absolutos $u_{\min}$ e $u_{\max}$ conteriam informações dos dados de teste.
2. O conjunto de treino estaria indiretamente "informado" sobre a dispersão futura dos dados não-vistos.

#### Protocolo Adotado no Projeto:
Ajustamos os limites $\min$ e $\max$ **estritamente sobre o conjunto de treino** ($N=353$):
$$u_{\min} = \min(u_{\text{treino}}), \quad u_{\max} = \max(u_{\text{treino}})$$
E aplicamos esses mesmos parâmetros fixos para transformar o teste:
$$u_{\text{norm}}^{(\text{teste})} = \frac{u_{\text{teste}} - u_{\min}}{u_{\max} - u_{\min}}$$

**Comprovação Numérica Obtida:**
* Intervalo no Treino: $[0{,}0000, 1{,}0000]$ (perfeitamente mapeado).
* Intervalo no Teste: $[-0{,}0043, 1{,}0388]$ (amostras de teste com IMC ligeiramente fora dos extremos do treino ultrapassam suavemente a fronteira, comprovando 100% de isolamento).

---

### 📚 5. Por que a RBF Gaussiana Exige Normalização de Escala?
A função de base radial gaussiana centrada em $c_j$ com dispersão $\sigma$ é dada por:
$$\phi_j(u) = \exp\left(-\frac{(u - c_j)^2}{2\sigma^2}\right)$$

1. **Evitar a Saturação Numérica:** Se a variável $u$ estivesse em sua magnitude física original (ex.: IMC variando entre $18$ e $42$), a distância quadrática $(u - c_j)^2$ atingiria centenas de unidades. Para qualquer valor modesto de $\sigma$, o expoente resultaria em $-\infty$, levando a $\phi_j(u) \approx 0$ (desativação das bases e anulação das colunas da matriz $\Phi$).
2. **Geometria dos Centros e Abertura:** Com o domínio restrito a $[0, 1]$, a distância entre $M$ centros uniformemente espaçados torna-se analiticamente simples:
   $$\Delta c = \frac{1}{M - 1}$$
   Isso permite calibrar a abertura de forma proporcional e elegante: $\sigma \approx \Delta c$, garantindo uma sobreposição balanceada entre gaussianas vizinhas sem risco de quase-colinearidade precoce.

---

### ❓ 21 Questões de Revisão Técnica — Dia 1 (Preparação para Defesa Oral)

1. **Qual a diferença fundamental entre Interpolação Numérica e Ajuste de Curvas por Regressão?**
   * *Resposta:* A interpolação exige que a curva passe exatamente por todos os pontos de dados ($f(x_i) = y_i$), sendo adequada quando não há ruído de medição. O ajuste de curvas busca minimizar uma função de perda global (como a soma dos quadrados dos resíduos), encontrando uma função suave que filtra o ruído e captura apenas a tendência do sistema físico.

2. **Por que o dataset `load_diabetes` é apropriado para cumprir as diretrizes do Projeto da Unidade 2?**
   * *Resposta:* Porque possui uma variável alvo contínua (progressão da doença) sujeita a ruídos biológicos e com dinâmica não-linear em relação aos biomarcadores, permitindo demonstrar a superioridade de bases locais (RBF) frente a modelos globais (MMQ).

3. **O que é vazamento de dados (*data leakage*) e como ele foi prevenido na etapa de normalização?**
   * *Resposta:* É a contaminação do processo de treinamento com informações estatísticas do conjunto de teste. Foi prevenido calculando-se os valores mínimo e máximo do IMC exclusivamente no conjunto de treino ($N=353$) e usando esses mesmos limites para projetar o teste.

4. **Por que as amostras de teste resultaram no intervalo $[-0{,}0043, 1{,}0388]$ e não estritamente em $[0, 1]$?**
   * *Resposta:* Porque o conjunto de teste continha pacientes com valores de IMC ligeiramente inferiores ao mínimo e superiores ao máximo observados no conjunto de treino. Esse comportamento é a prova matemática formal de que os dados de teste eram desconhecidos no momento do ajuste do escalonador.

5. **O que aconteceria matematicamente com a matriz de projeto $\Phi$ da RBF se a variável de entrada não fosse normalizada?**
   * *Resposta:* As distâncias euclidianas quadráticas $(u - c_j)^2$ seriam ordens de magnitude maiores que a abertura $\sigma$. A exponencial decairia para zero rapidamente, gerando colunas inteiras de zeros na matriz $\Phi$, tornando $\Phi^T \Phi$ singular e impossível de ser invertida de forma estável.

6. **Por que a divisão 80% treino e 20% teste é um padrão adotado na literatura?**
   * *Resposta:* O percentual de 80% (353 amostras) provê dados suficientes para que os pesos da RBF convirjam sem alta variância amostral, enquanto 20% (89 amostras) garante poder estatístico adequado para o cálculo confiável do EMQ e da correlação de Pearson sem subdimensionar o teste.

7. **Qual é o papel da semente pseudo-aleatória (`random_state=42`) na reprodutibilidade do projeto?**
   * *Resposta:* Fixa a sequência do gerador congruencial de números pseudo-aleatórios, garantindo que o particionamento das amostras seja idêntico em qualquer ambiente operacional, permitindo que os resultados numéricos possam ser auditados com precisão.

8. **Por que escolhemos a variável `bmi` (IMC) em vez de outras features disponíveis no dataset de diabetes?**
   * *Resposta:* O IMC é a variável isolada com maior correlação com a progressão da diabetes e possui dispersão contínua com variabilidade não-linear marcante, tornando evidente o contraste entre a reta do MMQ e a curva adaptativa da RBF.

9. **Como a normalização para o intervalo $[0, 1]$ simplifica a escolha dos hiperparâmetros $M$ e $\sigma$?**
   * *Resposta:* Torna o domínio da função previsível e independente de unidades físicas. Com isso, os centros podem ser alocados em posições conhecidas ($c_j \in [0, 1]$) e o espaçamento $\Delta c = 1/(M-1)$ dita diretamente a ordem de grandeza da largura $\sigma$.

10. **Se utilizássemos uma padronização Z-score ($\mu=0, \sigma=1$) em vez de MinMax, o que mudaria na formulação da RBF?**
    * *Resposta:* O domínio dos dados se estenderia de $-\infty$ a $+\infty$ (geralmente concentrado entre $[-3, +3]$). A distribuição dos centros precisaria cobrir esse intervalo estendido ou ser alocada com base em quantis/médias das amostras, tornando a heurística de centros uniformes menos direta do que no intervalo fechado $[0, 1]$.

11. **O que é ruído estocástico e como ele se manifesta neste projeto?**
    * *Resposta:* É uma variação aleatória, probabilística e não-determinística inerente aos dados reais ($y_i = f(u_i) + \epsilon_i$, onde $\epsilon_i \sim \mathcal{N}(0, \sigma_\epsilon^2)$). Na diabetes, reflete fatores biológicos não medidos (genética, dieta do dia) e variações laboratoriais. O modelo deve filtrar esse ruído em vez de interpolá-lo.

12. **O que são Splines Cúbicas e por que não as utilizamos neste projeto?**
    * *Resposta:* São métodos de interpolação polinomial por partes que garantem continuidade até a 2ª derivada nos nós (*knots*). Não são adequadas aqui porque são interpoladoras estritas (forçam a curva a passar por cada ponto), o que amplificaria o ruído biológico dos dados em vez de filtrar a tendência suave.

13. **Qual a diferença matemática fundamental entre Redes RBF e o algoritmo XGBoost?**
    * *Resposta:* A RBF gera uma curva contínua, analítica e infinitamente diferenciável ($\mathcal{C}^\infty$), treinada em uma única etapa pela Equação Normal via Álgebra Linear. O XGBoost é um ensemble de centenas de árvores de decisão que divide o espaço em degraus descontínuos e é treinado de forma iterativa por gradiente boosting.

14. **Nosso modelo utiliza apenas o IMC como entrada? Quais são as outras 9 variáveis do dataset?**
    * *Resposta:* Sim, utilizamos apenas o IMC (`bmi`) para modelar o ajuste de curva contínuo univariado $u \to y$, conforme o roteiro do professor. O dataset `load_diabetes` completo conta ainda com: idade (`age`), sexo (`sex`), pressão arterial (`bp`) e seis dosagens séricas: colesterol total (`s1`), LDL (`s2`), HDL (`s3`), razão colesterol/HDL (`s4`), triglicerídeos (`s5`) e glicose (`s6`).

15. **O que buscamos concretamente com o treinamento e modelagem deste projeto?**
    * *Resposta:* Modelar a dinâmica do sistema biológico através de uma função suave $\hat{y} = f(u)$, demonstrar formalmente a superioridade das bases locais da RBF frente à rigidez do MMQ tradicional e consolidar a implementação prática de estabilidade matricial via Álgebra Linear Numérica.

16. **O que é a Matriz de Projeto ($\Phi$) e qual o seu papel algébrico na RBF?**
    * *Resposta:* É a matriz $N \times (M+1)$ cujas colunas contêm o termo de viés (coluna de 1s) e a avaliação das $M$ funções gaussianas para cada amostra de treino. Ela transforma um problema não-linear nas entradas em um problema estritamente linear nos pesos ($\hat{\mathbf{y}} = \Phi \mathbf{w}$), permitindo solução analítica direta.

17. **O que é o Coeficiente de Correlação de Pearson ($r$) e como ele avalia o modelo?**
    * *Resposta:* É uma métrica estatística paramétrica adimensional que quantifica a força e a direção da associação linear entre os valores reais ($y$) e as estimativas do modelo ($\hat{y}$): $r = \frac{\text{cov}(y, \hat{y})}{\sigma_y \sigma_{\hat{y}}}$. Quanto mais próximo de $1{,}0$ no conjunto de teste, mais calibradas e proporcionais são as predições.

18. **Por que o conjunto de teste resultou no intervalo $[-0{,}0043, 1{,}0388]$ após o MinMaxScaler?**
    * *Resposta:* Porque o escalonador MinMax foi calibrado exclusivamente com o mínimo e máximo do conjunto de treino (evitando *data leakage*). Como o conjunto de teste possui pacientes com IMC ligeiramente fora dos extremos do treino, a projeção ultrapassou a faixa $[0, 1]$, comprovando cientificamente a ausência de vazamento de dados.

19. **O que é a abertura $\sigma$ da função gaussiana e como ela afeta geometricamente a curva?**
    * *Resposta:* É a largura (desvio-padrão) da função de base radial. Se for muito pequena ($\sigma \ll \Delta c$), gera "espinhos" pontiagudos isolados com vales artificiais entre centros; se for balanceada ($\sigma \approx \Delta c$), produz uma sobreposição suave; se for grande demais, achata as bases, torna as colunas de $\Phi$ colineares e explode o número de condição.

20. **O que significa alta variância amostral no contexto de ajuste de curvas?**
    * *Resposta:* É a hipersensibilidade do modelo à amostra de treino específica utilizada (sintoma de *overfitting*). Ocorre quando o modelo memoriza o ruído amostral, fazendo com que pequenas variações no conjunto de dados de entrada alterem drasticamente os pesos e a forma da curva ajustada.

21. **Qual a diferença entre os parâmetros ($\mathbf{w}$) e os hiperparâmetros ($M$ e $\sigma$)?**
    * *Resposta:* Os parâmetros $\mathbf{w} = [w_0, w_1, \dots, w_M]^T$ são os pesos aprendidos automaticamente pela resolução da Equação Normal. Os hiperparâmetros $M$ (quantidade de centros) e $\sigma$ (abertura das gaussianas) são decisões estruturais definidas pelo engenheiro antes do processo de ajuste.
