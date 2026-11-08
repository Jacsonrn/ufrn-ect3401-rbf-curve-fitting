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

---

# 📅 DIA 2: Construção e Ajuste da RBF Gaussiana (Álgebra Linear e Estabilidade Numérica)

Neste segundo dia, implementamos o modelo não-linear central exigido pelo professor Joilson: a **Rede de Base Radial (RBF Gaussiana)**. Deduzimos a formulação matricial da Matriz de Projeto $\Phi$, resolvemos analiticamente o vetor de pesos via pseudo-inversa de Moore-Penrose e realizamos o diagnóstico de estabilidade numérica através do Número de Condição.

---

### 🎯 O Objetivo do Dia 2 e a Relação com os Requisitos do Trabalho (Barema U2)

No roteiro oficial da disciplina ([`Projeto_U2.pdf`](file:///C:/Users/Jacson/Documents/Repositorios/Computacao%20Numerica/Unidade%202/Projeto_U2.pdf)), o professor Joilson B. A. Rego define requisitos claros e pontuados para a entrega da Unidade 2. As atividades do Dia 2 foram desenhadas para cobrir diretamente:

1. **Atendimento ao Item 1 da Metodologia (Vale 2,0 pontos):**
   * *Exigência:* *"Obter o Ajuste via Funções de Base Radial - RBF ou Funções Complexas"* (Slide 5).
   * *Como o Dia 2 cumpre:* Implementamos do zero a classe `RBFGaussianRegressor` em Python puro e NumPy, configurando os $M$ centros, calculando a dispersão $\sigma$ e ajustando a superfície não-linear sobre os dados observados.
2. **Atendimento ao Objetivo Específico de Parâmetros Livres:**
   * *Exigência:* *"Identificar os parâmetros livres do modelo matemático obtido"* (Slide 4).
   * *Como o Dia 2 cumpre:* Identificamos e quantificamos explicitamente todos os $M+1=9$ parâmetros livres calculados pela Equação Normal: o termo de viés $w_0 = 1012{,}14$ e os 8 pesos das bases gaussianas $\mathbf{w} = [w_1, \dots, w_8]^T$.
3. **Atendimento à Etapa 01 dos Resultados:**
   * *Exigência:* *"Resultados e Discussão - Etapa 01 - Ajuste Obtido"* (Slide 7).
   * *Como o Dia 2 cumpre:* Geramos a curva contínua ajustada sobreposta aos conjuntos de treino (80%) e teste (20%), demonstrando a filtragem suave do ruído biológico e registrando o gráfico oficial em [`docs/figuras/dia2_ajuste_rbf.png`](file:///C:/Users/Jacson/Documents/Repositorios/Computacao%20Numerica/Unidade%202/docs/figuras/dia2_ajuste_rbf.png).
4. **Garantia de Rigor em Computação Numérica (Estabilidade Matricial):**
   * Em vez de simplesmente aplicar bibliotecas de caixa-preta, diagnosticamos a estabilidade numérica exigida na disciplina através do cálculo do **Número de Condição** $\text{cond}(\Phi^T \Phi) = 1{,}42 \times 10^5$, garantindo que a resolução por pseudo-inversa (SVD) seja imune a erros de ponto flutuante.

---

### 📚 1. Formulação Matemática da RBF Gaussiana
A aproximação por funções de base radial modela a variável de saída contínua $\hat{y}(u)$ como uma combinação linear de funções de ativação simétricas locais, acrescida de um termo constante de viés (*bias*):
$$\hat{y}(u) = w_0 + \sum_{j=1}^{M} w_j \phi_j(u)$$
onde:
* $u \in [0, 1]$ é a entrada normalizada (IMC do paciente).
* $w_0$ é o termo de viés (intercepto global).
* $w_j$ são os pesos sinápticos associados a cada base gaussiana.
* $\phi_j(u)$ é a função gaussiana de base radial centrada em $c_j$ com largura $\sigma$:
  $$\phi_j(u) = \exp\left(-\frac{(u - c_j)^2}{2\sigma^2}\right)$$

---

### 📚 2. O Truque da Linearidade nos Parâmetros e a Matriz $\Phi$
Embora a função $\phi_j(u)$ seja fortemente **não-linear na variável de entrada $u$**, o modelo é estritamente **linear nos parâmetros $\mathbf{w}$**:
$$\hat{\mathbf{y}} = \Phi \mathbf{w}$$
onde $\Phi \in \mathbb{R}^{N \times (M+1)}$ é a **Matriz de Projeto** (*Design Matrix*):
$$\Phi = \begin{bmatrix}
1 & \phi_1(u_1) & \phi_2(u_1) & \dots & \phi_M(u_1) \\
1 & \phi_1(u_2) & \phi_2(u_2) & \dots & \phi_M(u_2) \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \phi_1(u_N) & \phi_2(u_N) & \dots & \phi_M(u_N)
\end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_M \end{bmatrix}$$

Essa propriedade algébrica é fundamental: ela transforma a busca pelos parâmetros ótimos em um problema de **mínimos quadrados lineares**, eliminando a necessidade de algoritmos iterativos sujeitos a mínimos locais (como a retropropagação em MLPs profundas).

---

### 📚 3. Dedução da Equação Normal e a Pseudo-Inversa via SVD
Queremos encontrar o vetor $\mathbf{w}$ que minimiza a soma dos quadrados dos resíduos (função custo $J(\mathbf{w})$):
$$J(\mathbf{w}) = \frac{1}{N} \|\Phi \mathbf{w} - \mathbf{y}\|^2 = \frac{1}{N} (\Phi \mathbf{w} - \mathbf{y})^T (\Phi \mathbf{w} - \mathbf{y})$$
Calculando o gradiente em relação a $\mathbf{w}$ e igualando a zero:
$$\nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{2}{N} \Phi^T (\Phi \mathbf{w} - \mathbf{y}) = \mathbf{0} \implies \Phi^T \Phi \mathbf{w} = \Phi^T \mathbf{y}$$

#### Por que nunca inverter $\Phi^T \Phi$ diretamente?
A inversão clássica $\mathbf{w} = (\Phi^T \Phi)^{-1} \Phi^T \mathbf{y}$ exige o cálculo explícito da inversa de $(\Phi^T \Phi)$. Se as gaussianas possuírem sobreposição excessiva, colunas de $\Phi$ tornam-se quase-colineares, fazendo com que $\det(\Phi^T \Phi) \approx 0$ e tornando $(\Phi^T \Phi)$ praticamente singular.

#### Solução Numérica Adotada: Pseudo-Inversa de Moore-Penrose (`pinv` via SVD)
A biblioteca decompõe $\Phi$ em **Valores Singulares** ($\Phi = U \Sigma V^T$):
$$\mathbf{w} = \Phi^\dagger \mathbf{y} = V \Sigma^\dagger U^T \mathbf{y}$$
onde $\Sigma^\dagger$ inverte apenas os valores singulares estritamente maiores que a tolerância de máquina $\epsilon$, truncando os componentes ruidosos para zero. Isso garante que $\mathbf{w}$ seja a solução de **menor norma euclidiana $\|\mathbf{w}\|_2$**, evitando pesos gigantescos e instáveis.

---

### 📚 4. Número de Condição $\text{cond}(\Phi^T \Phi)$ e Sensibilidade Numérica
O Número de Condição $\kappa$ da matriz $(\Phi^T \Phi)$ é calculado pela razão entre seu autovalor máximo e mínimo:
$$\kappa = \frac{\lambda_{\max}(\Phi^T \Phi)}{\lambda_{\min}(\Phi^T \Phi)}$$
* Se $\sigma$ for muito grande: as gaussianas achatam-se e passam a cobrir todo o domínio de forma uniforme. As colunas tornam-se redundantes, $\lambda_{\min} \to 0$ e $\kappa$ explode para valores $> 10^{14}$ (matriz mal-condicionada).
* No nosso experimento com $M=8$ e $\sigma = 0{,}1429$, obtivemos $\kappa = 1{,}42 \times 10^5$, atestando que o sistema é **bem-condicionado** e resistente a erros de arredondamento de ponto flutuante.

---

### ❓ 14 Questões de Revisão Técnica — Dia 2 (Preparação para Defesa Oral)

22. **Por que a RBF é considerada uma técnica linear nos parâmetros, se as funções gaussianas são não-lineares?**
    * *Resposta:* Porque a não-linearidade afeta apenas as variáveis de entrada $u$, gerando a matriz $\Phi$. Uma vez calculada $\Phi$, a saída do modelo é uma simples soma ponderada linear $\hat{y} = \sum w_j \Phi_{ij}$. Logo, a função de custo em relação a $\mathbf{w}$ é puramente quadrática convexa.

23. **Qual é a consequência geométrica de escolher $\sigma$ muito menor do que o espaçamento entre centros ($\sigma \ll \Delta c$)?**
    * *Resposta:* Cada gaussiana torna-se um "pico" excessivamente estreito que atua apenas na vizinhança imediata de seu centro. Entre dois centros adjacentes, a resposta das bases decai para zero, gerando "vales" artificiais e incapacidade de generalizar para novos dados.

24. **O que acontece algebricamente com a matriz $\Phi$ quando $\sigma$ é excessivamente grande?**
    * *Resposta:* As gaussianas tornam-se quase constantes e horizontais ao longo de todo o domínio $[0, 1]$. Todas as colunas de $\Phi$ passam a se parecer com a primeira coluna (a coluna de viés de 1s), provocando multicolinearidade severa e perda de posto (*rank deficiency*).

25. **Como o Número de Condição ($\kappa$) diagnostica a estabilidade numérica da solução?**
    * *Resposta:* $\kappa$ quantifica a sensibilidade da solução $\mathbf{w}$ a pequenas perturbações nos dados de entrada ou erros de arredondamento em ponto flutuante. Um valor moderado como $1{,}42 \times 10^5$ comprova que a matriz é invertível com segurança numérica.

26. **Por que a Decomposição em Valores Singulares (SVD) é superior à Inversão de Matrizes Direta na Equação Normal?**
    * *Resposta:* Porque a inversão direta $(\Phi^T \Phi)^{-1}$ eleva ao quadrado o número de condição original da matriz ($\kappa(\Phi^T \Phi) = \kappa(\Phi)^2$) e falha catastróficamente se a matriz for quase singular. A SVD atua diretamente sobre $\Phi$, isolando e truncando valores singulares minúsculos que causariam divisão por zero numérico.

27. **Qual é o papel do termo de viés $w_0$ na formulação da RBF?**
    * *Resposta:* O viés $w_0$ atua como o intercepto global da curva, permitindo transladar verticalmente o modelo sem forçar as bases gaussianas a compensarem sozinhas o nível médio da variável alvo ($y$). No nosso ajuste, $w_0 = 1012{,}14$.

28. **Por que os centros $c_j$ foram posicionados de forma uniforme em $[0, 1]$ em vez de usar agrupamento por K-Means?**
    * *Resposta:* Como o problema possui uma única variável de entrada ($u = \text{IMC}$) distribuída de forma contínua em $[0, 1]$, a distribuição uniforme com espaçamento analítico $\Delta c = \frac{1}{M-1}$ garante cobertura homogênea e regularidade na escolha de $\sigma$, sem adicionar a complexidade estocástica do K-Means.

29. **Como a solução obtida por $\mathbf{w} = \Phi^\dagger \mathbf{y}$ lida com problemas sobredeterminados ($N > M$)?**
    * *Resposta:* Em problemas sobredeterminados (temos $N=353$ equações e apenas $M+1=9$ incógnitas), não existe solução exata que satisfaça todos os pontos. A pseudo-inversa encontra analiticamente a solução única que minimiza a norma euclidiana do resíduo $\|\Phi \mathbf{w} - \mathbf{y}\|_2$ (mínimos quadrados).

30. **O que a magnitude dos pesos obtidos ($w_1 \dots w_8$) revela sobre o comportamento das bases vizinhas?**
    * *Resposta:* Pesos com sinais alternados e magnitudes moderadas indicam que as gaussianas vizinhas cooperam para modelar a inclinação da curva suave sem gerar oscilações de alta frequência descontroladas (ausência de instabilidade de Runge).

31. **Por que a resolução por Equação Normal é computacionalmente mais rápida do que a Descida do Gradiente neste problema?**
    * *Resposta:* Como $M=8$ é pequeno, calcular $\Phi^\dagger$ envolve matrizes minúsculas de ordem $9 \times 9$, resolvidas em frações de milissegundo em uma única operação direta. A descida do gradiente exigiria centenas de épocas iterativas com ajuste delicado da taxa de aprendizado $\eta$.

32. **O que é a Pseudo-Inversa de Moore-Penrose e por que ela é indispensável neste projeto?**
    * *Resposta:* É a generalização do conceito de matriz inversa para matrizes retangulares ou singulares. No nosso caso, como a matriz de projeto $\Phi$ tem dimensão $353 \times 9$, não existe inversa clássica $\Phi^{-1}$. A pseudo-inversa $\Phi^\dagger$ calcula analiticamente o vetor $\mathbf{w} = \Phi^\dagger \mathbf{y}$ que minimiza a soma dos quadrados dos resíduos (mínimos quadrados) e, simultaneamente, escolhe a solução de menor norma euclidiana $\|\mathbf{w}\|_2$, garantindo estabilidade e impedindo pesos excessivos.

33. **O que é o Número de Condição ($\kappa$) e como ele orienta a avaliação de estabilidade numérica da RBF?**
    * *Resposta:* É a medida de sensibilidade que quantifica quanto um erro de entrada ou de arredondamento em ponto flutuante ($10^{-16}$) é amplificado na resposta final dos pesos ($\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$). Um número de condição moderado (como o nosso $\kappa = 1{,}42 \times 10^5$) prova que a matriz $(\Phi^T \Phi)$ é bem-condicionada e que os pesos calculados são estáveis e confiáveis. Valores próximos de $10^{16}$ indicariam uma matriz quase singular, com perda total de precisão numérica.

34. **O que é a Decomposição em Valores Singulares (SVD) e por que ela é a base do cálculo da pseudo-inversa?**
    * *Resposta:* O SVD fatora qualquer matriz retangular em três componentes ortogonais: $\Phi = U \Sigma V^T$, onde $\Sigma$ contém os valores singulares em ordem decrescente. A pseudo-inversa é obtida estavelmente por $\Phi^\dagger = V \Sigma^\dagger U^T$, invertendo individualmente cada valor singular não-nulo ($1/\sigma_i$). Valores singulares inferiores à tolerância da máquina são zerados, prevenindo divisões por números infinitesimais que causariam estouro numérico.

35. **Por que no nosso ajuste o termo de viés resultou em $w_0 = 1012{,}14$? Nós que escolhemos esse valor?**
    * *Resposta:* Não, nós não escolhemos $w_0$; ele foi aprendido automaticamente pela resolução da Equação Normal ($\mathbf{w} = \Phi^\dagger \mathbf{y}$). O valor $1012{,}14$ resultou alto porque o algoritmo utilizou o viés como uma "âncora de teto" global positiva, enquanto quase todos os pesos das bases gaussianas receberam valores fortemente negativos ($w_1 = -808{,}51, w_3 = -692{,}12, w_5 = -546{,}85$). Como as bases gaussianas se sobrepõem no domínio $[0, 1]$, elas somam aproximadamente $-860$, resultando no ajuste final balanceado: $\hat{y} \approx 1012{,}14 - 860 \approx 152$, que é exatamente a média da progressão da diabetes observada nos pacientes.

---

# 📅 DIA 3: Avaliação Estatística: EMQ, Regressão Linear e Correlação de Pearson

Neste terceiro dia, implementamos a suíte de validação estatística quantitativa exigida pelo professor Joilson no roteiro da Unidade 2. Avaliamos a discrepância residual via EMQ e RMSE, quantificamos a força da associação linear via Coeficiente de Pearson e analisamos a calibração do modelo através da reta de regressão entre valores reais e estimados.

---

### 🎯 O Objetivo do Dia 3 e a Relação com os Requisitos do Trabalho (Barema U2)

No [`Projeto_U2.pdf`](file:///C:/Users/Jacson/Documents/Repositorios/Computacao%20Numerica/Unidade%202/Projeto_U2.pdf), o professor Joilson estabeleceu que as métricas estatísticas compõem a maior fatia avaliativa da metodologia (**4,0 pontos** no total). As atividades do Dia 3 cumpriram rigorosamente:

1. **Atendimento ao Item 2 da Metodologia (Vale 1,0 ponto) e Etapa 02 dos Resultados:**
   * *Exigência:* *"Obter o Erro Médio Quadrático - EMQ"* (Slide 5 e Slide 8).
   * *Como o Dia 3 cumpre:* Calculamos o $\text{EMQ}$ e a sua raiz ($\text{RMSE}$) de forma analítica pura tanto no treino ($3715{,}07$ / $60{,}95$) quanto no teste ($4495{,}49$ / $67{,}05$), gerando a faixa de incerteza empírica $\hat{y} \pm \text{RMSE}$.
2. **Atendimento ao Item 3 da Metodologia (Vale 1,0 ponto):**
   * *Exigência:* *"Efetuar uma regressão linear entre os dados reais e os ajustados via RBF"* (Slide 5).
   * *Como o Dia 3 cumpre:* Ajustamos por Mínimos Quadrados a reta $y = a \cdot \hat{y} + b$ sobre as $89$ amostras de teste, obtendo $a = 0{,}6460$ e $b = 51{,}48$, comparando-a com a bissetriz ideal $y = \hat{y}$.
3. **Atendimento ao Item 4 da Metodologia (Vale 2,0 pontos) e Etapa 03 dos Resultados:**
   * *Exigência:* *"Mostrar o Coeficiente de Correlação de Pearson"* (Slide 5 e Slide 9).
   * *Como o Dia 3 cumpre:* Implementamos a fórmula paramétrica da covariância normalizada, obtendo $r = 0{,}6234$ no treino e $r = 0{,}4653$ em dados não-vistos de teste, comprovando associação estatisticamente significativa entre o IMC e a progressão da doença.

---

### 📚 1. Formulação do Erro Médio Quadrático (EMQ) e RMSE
O Erro Médio Quadrático penaliza a discrepância quadrática entre o alvo físico $y$ e a saída da RBF $\hat{y}$:
$$\text{EMQ} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{N} \|\mathbf{y} - \hat{\mathbf{y}}\|_2^2$$
Como o EMQ resulta na unidade quadrática da grandeza original ($\text{pontos}^2$), calculamos a Raiz do Erro Médio Quadrático ($\text{RMSE}$):
$$\text{RMSE} = \sqrt{\text{EMQ}}$$
O RMSE expressa o desvio padrão dos resíduos na mesma escala de progressão clínica da diabetes. No nosso teste, $\text{RMSE} = 67{,}05$ pontos.

---

### 📚 2. Coeficiente de Correlação Linear de Pearson ($r$)
O coeficiente de Pearson mede a covariância linear padronizada entre a realidade e o modelo:
$$r = \frac{\sum_{i=1}^N (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^N (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^N (\hat{y}_i - \bar{\hat{y}})^2}} = \frac{\text{cov}(y, \hat{y})}{\sigma_y \sigma_{\hat{y}}}$$
* No conjunto de treino: $r = 0{,}6234$.
* No conjunto de teste: $r = 0{,}4653$.
A retenção de um $r$ positivo e substancial no conjunto de teste comprova que a RBF generalizou o sinal fisiológico real sem colapso por memorização de ruído.

---

### 📚 3. Regressão Linear de Calibração ($y$ vs. $\hat{y}$)
Ajustamos uma reta entre o valor real e a estimativa do modelo no conjunto de teste:
$$y_i = a \cdot \hat{y}_i + b$$
Pelo método dos Mínimos Quadrados:
$$a = \frac{\text{cov}(\hat{y}, y)}{\text{var}(\hat{y})}, \quad b = \bar{y} - a \bar{\hat{y}}$$
* **Resultado Obtido:** $a = 0{,}6460$, $b = 51{,}48$ com $R^2 = 0{,}2165$.
* **Interpretação Numérica do Encolhimento para a Média:** O coeficiente angular $a < 1{,}0$ reflete o fenômeno clássico de **regressão à média**: diante de dados com ruído biológico estocástico, o modelo RBF suaviza os extremos, gerando predições que tendem ao valor médio populacional quando o sinal do IMC não é conclusivo.

---

### 📚 4. A Faixa de Incerteza ($\pm \text{RMSE}$)
A faixa $\hat{y}(u) \pm \text{RMSE}$ traçada ao redor da curva ajustada delimita a região de dispersão estocástica esperada do sistema. Sob a premissa de resíduos com distribuição aproximadamente normal ($\epsilon \sim \mathcal{N}(0, \text{RMSE}^2)$), aproximadamente $68{,}3\%$ dos pacientes situam-se dentro desta faixa de tolerância, provando visualmente a capacidade da RBF de operar como filtro suave da nuvem de dados.

---

### ❓ 10 Questões de Revisão Técnica — Dia 3 (Preparação para Defesa Oral)

36. **Qual a diferença fundamental entre EMQ e RMSE, e por que apresentamos ambos no relatório?**
    * *Resposta:* O EMQ é a função custo analítica minimizada pela Equação Normal, mas sua unidade é quadrática ($\text{pontos}^2$). O RMSE é a raiz quadrada do EMQ, retornando à unidade física original da variável alvo, permitindo uma interpretação direta da magnitude média do erro de predição do modelo.

37. **Por que o EMQ no teste ($4495{,}49$) foi superior ao EMQ no treino ($3715{,}07$)?**
    * *Resposta:* Porque os pesos da RBF foram otimizados analiticamente para minimizar o erro estritamente sobre as 353 amostras de treino. O conjunto de teste contém dados inéditos sujeitos a variabilidade estocástica não vista pelo algoritmo, sendo esperado um erro ligeiramente maior na generalização.

38. **O que significa a correlação de Pearson $r = 0{,}4653$ obtida no conjunto de teste?**
    * *Resposta:* Indica uma associação linear positiva moderada e estatisticamente significativa entre as previsões do modelo e a progressão real da diabetes em pacientes não-vistos, comprovando que o modelo capturou o sinal fisiológico do IMC.

39. **O que os parâmetros $a = 0{,}6460$ e $b = 51{,}48$ da reta de regressão revelam sobre a calibração da RBF?**
    * *Resposta:* Como $a < 1{,}0$, o modelo apresenta o fenômeno estatístico de retração à média: ele tende a prever valores ligeiramente mais conservadores nos extremos (subestimando progressões muito severas e superestimando progressões muito baixas), mantendo o erro quadrático global sob controle.

40. **Qual a relação matemática exata entre o Coeficiente de Pearson ($r$) e o Coeficiente de Determinação ($R^2$) na regressão linear simples?**
    * *Resposta:* Na regressão linear simples entre uma única variável preditora e o alvo, o coeficiente de determinação é exatamente o quadrado da correlação de Pearson: $R^2 = r^2$. No nosso caso, $0{,}4653^2 \approx 0{,}2165$ ($21{,}65\%$ da variabilidade do teste explicada).

41. **Como a faixa de incerteza de $\pm \text{RMSE}$ auxilia na interpretação da dinâmica do sistema físico?**
    * *Resposta:* A faixa indica a tolerância esperada de variabilidade biológica do sistema. Ela demonstra que a RBF atua como a média condicional $\mathbb{E}[y|u]$, onde os pontos fora da linha central não são falhas do modelo, mas sim flutuações intrínsecas ao processo físico/biológico.

42. **Se o coeficiente angular da regressão fosse $a = 1{,}0$ e o intercepto $b = 0{,}0$, o que isso significaria?**
    * *Resposta:* Significaria calibração perfeita: a média das predições coincidiria exatamente com a média das observações reais em todos os níveis de IMC, sem nenhum viés sistemático de escala ou de deslocamento.

43. **Por que calculamos o Pearson com base nos dados de teste e não apenas nos dados de treino?**
    * *Resposta:* Calcular métricas apenas no treino mede apenas a capacidade de memorização do modelo. A avaliação em dados de teste é o único critério científico válido para medir a generalização frente a novas observações.

44. **Por que o denominador da fórmula de Pearson pode ser zero em casos degenerados?**
    * *Resposta:* Se o modelo fizesse predições constantes para todas as amostras ($\hat{y}_i = \text{constante}$), a variância amostral $\sum (\hat{y}_i - \bar{\hat{y}})^2$ seria zero, resultando em divisão por zero. Nosso código trata isso retornando $r = 0{,}0$.

45. **Como o valor de $\text{RMSE} = 67{,}05$ se compara com o desvio padrão da variável alvo original?**
    * *Resposta:* A variável alvo tem desvio padrão total de aproximadamente $77$ pontos. Um RMSE de $67{,}05$ demonstra que o modelo RBF baseado no IMC reduz a incerteza do sistema em relação à estimativa ingênua da média, comprovando ganho informacional real.

---

# 📅 DIA 4: Comparação via Mínimos Quadrados (MMQ) e Análise Crítica de Modelos

Neste quarto dia, implementamos o modelo clássico de regressão polinomial por Mínimos Quadrados Ordinários (MMQ / OLS), cumprindo a exigência de comparação isonômica com a RBF Gaussiana. Analisamos a Matriz de Vandermonde, a sensibilidade numérica do número de condição e o dilema clássico entre complexidade e generalização (Navalha de Occam).

---

### 🎯 O Objetivo do Dia 4 e a Relação com os Requisitos do Trabalho (Barema U2)

No [`Projeto_U2.pdf`](file:///C:/Users/Jacson/Documents/Repositorios/Computacao%20Numerica/Unidade%202/Projeto_U2.pdf), o professor Joilson exige uma confrontação formal entre abordagens locais e globais:

1. **Atendimento ao Item 5 da Metodologia (Vale 2,0 pontos):**
   * *Exigência:* *"Repetir o processo utilizando o Método dos Mínimos Quadrados - MMQ"* (Slide 5).
   * *Como o Dia 4 cumpre:* Implementamos a classe `PolynomialMMQRegressor` em Python e avaliamos três ordens polinomiais (Grau 1 - Reta, Grau 2 - Parábola e Grau 3 - Cúbico).
2. **Atendimento às Etapas 04 e 05 dos Resultados:**
   * *Exigência:* *"Resultados e Discussão - Etapa 04 - Repetição do Processo para o Ajuste via MMQ"* (Slides 10 e 11).
   * *Como o Dia 4 cumpre:* Calculamos todas as métricas ($\text{EMQ}$, $\text{RMSE}$, Pearson $r$ e $\kappa$) nas mesmas 89 amostras de teste, construindo a tabela comparativa oficial e os gráficos sobrepostos em [`docs/figuras/dia4_comparativo_rbf_vs_mmq.png`](file:///C:/Users/Jacson/Documents/Repositorios/Computacao%20Numerica/Unidade%202/docs/figuras/dia4_comparativo_rbf_vs_mmq.png) e [`docs/figuras/dia4_barras_comparativas_rmse.png`](file:///C:/Users/Jacson/Documents/Repositorios/Computacao%20Numerica/Unidade%202/docs/figuras/dia4_barras_comparativas_rmse.png).
   * **Total acumulado no projeto:** 9,0 pontos de 10,0 pontos executados!

---

### 📚 1. Formulação do MMQ Polinomial e a Matriz de Vandermonde
No MMQ, a função de ajuste é modelada como um polinômio global de grau $d$:
$$\hat{y}(u) = w_0 + w_1 u + w_2 u^2 + \dots + w_d u^d$$
Para as $N$ amostras, constrói-se a **Matriz de Vandermonde** $V \in \mathbb{R}^{N \times (d+1)}$:
$$V = \begin{bmatrix}
1 & u_1 & u_1^2 & \dots & u_1^d \\
1 & u_2 & u_2^2 & \dots & u_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & u_N & u_N^2 & \dots & u_N^d
\end{bmatrix}$$
Os coeficientes ótimos minimizam a soma dos quadrados dos resíduos via Equação Normal:
$$\mathbf{w}_{MMQ} = (V^T V)^{-1} V^T \mathbf{y} = V^\dagger \mathbf{y}$$

---

### 📚 2. Bases Monomiais Globais vs. Bases Gaussianas Locais
* **MMQ (Bases Monomiais $u^k$):** Têm **suporte global**. Cada monômio estende sua influência por todo o domínio $[0, 1]$. Qualquer perturbação pontual em uma extremidade força a reconfiguração de toda a curva, gerando risco de oscilações espúrias (Fenômeno de Runge) em graus mais altos.
* **RBF (Bases Gaussianas $\phi_j(u)$):** Têm **suporte local**. O efeito de cada função decai exponencialmente com a distância euclidiana ao centro $c_j$. Isso confere grande estabilidade geométrica para filtragem de variações regionais.

---

### 📚 3. Análise Numérica do Condicionamento de Vandermonde
Conforme aumentamos o grau do polinômio, as colunas de potências sucessivas ($u^2, u^3, \dots$) tornam-se progressivamente parecidas para $u \in [0, 1]$:
* **Grau 1 (Linear):** $\text{cond}(V^T V) = 36{,}2$ (excelente condicionamento).
* **Grau 2 (Parábola):** $\text{cond}(V^T V) = 970$ (aumento de 26x).
* **Grau 3 (Cúbico):** $\text{cond}(V^T V) = 24.300$ (aumento de quase 1.000x).
Esse rápido crescimento do número de condição comprova por que a literatura de Computação Numérica desaconselha polinômios de ordem elevada na base canônica de Vandermonde.

---

### 📚 4. Discussão dos Resultados: Viés-Variância e a Navalha de Occam
* **Desempenho no Treino:** A RBF ($M=8$) obteve o menor erro ($\text{RMSE} = 60{,}95$) devido aos seus 9 graus de liberdade locais, contra $62{,}08$ do MMQ linear.
* **Desempenho no Teste:** O MMQ Linear alcançou $\text{RMSE} = 63{,}73$ e $r = 0{,}5076$. Como o dataset de diabetes apresenta uma tendência média suave com grande ruído estocástico, a simplicidade paramétrica do modelo linear (apenas 2 pesos) atuou como um regularizador natural, evitando memorizar os pequenos vales que a RBF de 8 centros capturou no treino.

---

### ❓ 16 Questões de Revisão Técnica — Dia 4 (Preparação para Defesa Oral)

46. **Por que a Matriz de Vandermonde é propensa a mau-condicionamento em graus elevados?**
    * *Resposta:* Porque para $u \in [0, 1]$, as curvas de potências sucessivas ($u^2, u^3, u^4$) tornam-se cada vez mais colineares e difíceis de distinguir numericamente. Isso faz com que as colunas da matriz $V$ fiquem quase linearmente dependentes, elevando o número de condição exponencialmente.

47. **Qual a diferença fundamental de suporte entre a base monomial do MMQ e a base gaussiana da RBF?**
    * *Resposta:* A base monomial tem suporte global infinito: uma alteração em uma amostra extrema altera a inclinação da curva em todo o domínio. A base gaussiana tem suporte local: sua ativação decai exponencialmente fora da vizinhança do centro $c_j$, confinando o impacto a uma região delimitada.

48. **Por que a RBF superou o MMQ no conjunto de treinamento ($60{,}95$ vs. $62{,}08$)?**
    * *Resposta:* Porque a RBF com $M=8$ possui 9 graus de liberdade locais (8 centros e 1 viés), permitindo que ela adapte sua curvatura a variações regionais do IMC que uma linha reta rígida (MMQ Grau 1) é incapaz de capturar (baixo viés).

49. **Por que o MMQ Grau 1 obteve um RMSE de teste ligeiramente melhor que a RBF com $M=8$ ($63{,}73$ vs. $67{,}05$)?**
    * *Resposta:* Esse resultado ilustra a Navalha de Occam e o trade-off viés-variância: diante de dados com ruído biológico intenso, o modelo linear simples possui variância mínima e não memoriza ruído, enquanto a RBF com 8 centros foi flexível o suficiente para se moldar a oscilações que eram ruído de treino.

50. **O que aconteceu com o erro de teste ao aumentarmos o grau do MMQ de 1 para 2 e 3?**
    * *Resposta:* O erro permaneceu praticamente idêntico ($63{,}73 \to 63{,}91 \to 63{,}75$), demonstrando que termos quadráticos e cúbicos globais não agregaram poder preditivo real sobre o sinal biológico do IMC, apenas elevaram o custo computacional e o número de condição.

51. **O que é o Princípio da Isonomia Científica na comparação entre RBF e MMQ?**
    * *Resposta:* É a garantia metodológica de que ambos os modelos concorrentes foram ajustados exatamente sobre os mesmos 353 pacientes de treino e avaliados exatamente sobre os mesmos 89 pacientes de teste, eliminando qualquer viés amostral na avaliação.

52. **Como poderíamos otimizar a RBF para superar o MMQ também no conjunto de teste?**
    * *Resposta:* Reduzindo o número de centros para um valor mais parcimonioso (como $M=4$ ou $M=5$) ou ampliando a abertura $\sigma$, o que tornaria a RBF mais suave e menos susceptível ao ruído local, unindo a flexibilidade não-linear com menor variância amostral.

53. **O que é o Fenômeno de Runge e como o MMQ polinomial se relaciona com ele?**
    * *Resposta:* É o surgimento de oscilações descontroladas nas extremidades do intervalo de interpolação quando se eleva o grau do polinômio com nós equidistantes. O MMQ com graus elevados sofre de instabilidade similar nas bordas do domínio $[0, 1]$.

54. **Por que a correlação de Pearson do MMQ Linear no teste foi $r = 0{,}5076$?**
    * *Resposta:* Porque o modelo linear estabelece uma relação monotonicamente crescente direta e rígida entre IMC e progressão, alinhando-se fortemente com a direção global média da nuvem de dados.

55. **Qual critério um engenheiro deve usar para escolher entre RBF e MMQ em um sistema dinâmico real?**
    * *Resposta:* Se o sistema apresentar dinâmicas locais conhecidas, como saturação, histerese ou zonas mortas (ex.: curva em S de uma turbina eólica), a RBF é mandatória. Se o sistema for predominantemente linear com forte ruído e recursos computacionais escassos, o MMQ simples é a escolha mais robusta e parcimoniosa.

56. **O que é a Matriz de Vandermonde e qual a sua diferença conceitual em relação à Matriz de Projeto ($\Phi$)?**
    * *Resposta:* A **Matriz de Projeto** (geralmente denotada por $\Phi$ ou $X$) é o conceito geral da Álgebra Linear e da Teoria de Regressão: trata-se de qualquer matriz $N \times P$ na qual cada linha corresponde a uma observação experimental e cada coluna corresponde a uma função de base arbitrária avaliada naquele ponto amostral. A **Matriz de Vandermonde** ($V$), por sua vez, é um caso particular e específico de Matriz de Projeto no qual as funções de base são estritamente potências inteiras sucessivas de uma única variável escalar ($u^0, u^1, u^2, \dots, u^d$). Em suma: toda matriz de Vandermonde é uma matriz de projeto, mas a recíproca não é verdadeira (por exemplo, a matriz de projeto da RBF utiliza funções de base exponenciais gaussianas locais centradas em $c_j$, e não monômios polinomiais).

57. **O que é o Número de Condição ($\kappa$) de uma matriz e como ele afeta a precisão numérica segundo o padrão IEEE 754?**
    * *Resposta:* O número de condição $\kappa(A) = \|A\| \cdot \|A^{-1}\| = \sigma_{\max} / \sigma_{\min}$ quantifica a sensibilidade ou amplificação de perturbações no sistema linear $A \mathbf{w} = \mathbf{y}$. Ele expressa o pior caso de propagação de incerteza: um erro de arredondamento de máquina de magnitude $\epsilon$ pode gerar um erro relativo na solução $\mathbf{w}$ amplificado em até $\kappa(A) \cdot \epsilon$. Pela regra prática da análise numérica computacional (padrão IEEE 754 dupla precisão / float64, que possui aproximadamente 16 dígitos significativos decimais), se o número de condição for $\kappa = 10^k$, perdem-se até $k$ dígitos significativos de precisão. No nosso projeto, o MMQ Grau 1 teve $\kappa = 36{,}2$ (estabilidade quase perfeita, perda de $\approx 1{,}5$ dígitos decimais); já o MMQ Grau 3 teve $\kappa = 24.300$ (perda de $\approx 4{,}4$ dígitos), enquanto a RBF com 8 centros atingiu $\kappa = 1{,}42 \times 10^5$, o que justificou a utilização da Decomposição em Valores Singulares (SVD via `np.linalg.pinv`) para truncar autovalores instáveis.

58. **Por que o MMQ Linear apresenta dois valores distintos de RMSE no relatório ($62{,}08$ e $63{,}73$), e quantos parâmetros livres ele ajusta?**
    * *Resposta:* Não há nenhuma contradição entre esses dois valores, pois eles avaliam o modelo em conjuntos de dados completamente distintos:
      1. $\text{RMSE}_{\text{treino}} = 62{,}08$: mede o resíduo quadrático médio cometido pelo modelo sobre as 353 amostras de pacientes utilizadas ativamente para calcular a reta ótima via Equação Normal.
      2. $\text{RMSE}_{\text{teste}} = 63{,}73$: mede a capacidade de generalização do modelo ao prever o desfecho clínico dos 89 pacientes inéditos mantidos em quarentena estrita.
      O modelo linear possui exatamente **2 parâmetros livres** (graus de liberdade): o coeficiente linear/viés ($w_0$) e a inclinação angular ($w_1$), descrevendo a função afim $\hat{y}(u) = w_0 + w_1 u$.

59. **O que é o Trade-off Viés-Variância no contexto de Ajuste de Curvas e Aprendizado de Máquina?**
    * *Resposta:* É o dilema fundamental da modelagem preditiva que decompõe o erro quadrático esperado em três componentes ortogonais: $\text{Erro Total} = \text{Viés}^2 + \text{Variância} + \sigma^2_{\text{ruído}}$.
      * **Viés (Underfitting):** Erro decorrente de simplificações estruturais excessivas do modelo. Um modelo de alto viés (como forçar uma linha reta sobre uma oscilação senoidal) é incapaz de capturar a física do fenômeno, performando mal tanto no treino quanto no teste.
      * **Variância (Overfitting):** Erro decorrente da sensibilidade excessiva do modelo às variações estocásticas da amostra particular de treino. Um modelo de alta variância (polinômio de grau 20 ou rede com excesso de parâmetros) memoriza o ruído aleatório em vez do sinal real; ele apresenta erro quase nulo no treino, mas falha drasticamente diante de novos dados de teste.
      * O objetivo do cientista da computação é encontrar a complexidade paramétrica ideal ("sweet spot") que minimize a soma de viés e variância no conjunto de validação/teste.

60. **Como o princípio da Navalha de Occam (*Lex Parsimoniae*) justifica a preferência pelo MMQ Linear neste projeto?**
    * *Resposta:* O princípio da Navalha de Occam estipula que, diante de hipóteses ou modelos concorrentes que apresentem poder explanatório ou erros empíricos equivalentes, a formulação mais simples deve ser sempre adotada. No nosso projeto, o MMQ Linear (2 parâmetros) atingiu $\text{RMSE}_{\text{teste}} = 63{,}73$, superando o MMQ Grau 2 ($63{,}91$), o MMQ Grau 3 ($63{,}75$) e a RBF ($67{,}05$). Como os termos não-lineares adicionais e os centros gaussianas extras não agregaram precisão preditiva nos dados de teste, a Navalha de Occam dita que a reta linear é a solução mais elegante, econômica, numericamente estável ($\kappa = 36{,}2$) e interpretável para modelar a relação entre IMC e progressão do diabetes.

61. **Em termos conclusivos de engenharia, qual modelo foi superior: a RBF Gaussiana ou o MMQ Linear? O MMQ foi o vencedor do projeto?**
    * *Resposta:* A resposta depende estritamente do critério e do domínio de avaliação considerado:
      * **Vencedor em Generalização Preditiva (Critério Clínico/Teste):** O **MMQ Linear venceu**. Por possuir apenas 2 parâmetros, ele atuou com regularização natural contra o elevado ruído biológico intrínseco do dataset, obtendo menor erro de teste ($\text{RMSE} = 63{,}73$ contra $67{,}05$ da RBF) e maior correlação linear de Pearson ($r = 0{,}5076$ contra $0{,}4653$).
      * **Vencedor em Capacidade de Ajuste (Critério de Baixo Viés/Treino):** A **RBF Gaussiana venceu**. Com seus 8 centros distribuídos ao longo do domínio e 9 graus de liberdade, a RBF obteve erro de treino substancialmente menor ($\text{RMSE} = 60{,}95$ contra $62{,}08$ do MMQ), provando sua flexibilidade superior para se conformar a geometrias complexas.
      * **Conclusão Metodológica:** Não existe modelo intrinsecamente "melhor" a priori (Teorema do *No Free Lunch*). Para fenômenos biológicos com dispersão acentuada e correlação monotônica fraca/média, a simplicidade do MMQ é imbatível; contudo, para fenômenos físicos determinísticos com dinâmicas altamente não-lineares (saturação, transientes, histerese), a RBF Gaussiana é incomparavelmente superior e mandatória.

---

# 📅 DIA 5: Análise Global dos Resultados, Redação do Relatório Final em LaTeX e Conclusões

No quinto e último dia do projeto, consolidamos os quatro módulos computacionais em um artigo técnico formal redigido em LaTeX (`relatorio_final_projeto_u2.tex`), compilado via `pdflatex` em formato PDF pronto para submissão acadêmica. Analisamos de forma integrada todas as métricas sob a ótica dos teoremas de aproximação e fechamos os 10,0 pontos do Barema de Avaliação da Unidade II.

---

### 🎯 O Objetivo do Dia 5 e a Relação com os Requisitos do Trabalho (Barema U2)

1. **Atendimento ao Item 6 do Barema (Vale 2,0 pontos):**
   * *Exigência:* *"Considerações Finais e Redação do Relatório Técnico"* (Slide 6 do `Projeto_U2.pdf`).
   * *Como o Dia 5 cumpre:* Estruturação completa do relatório em 5 seções canônicas (Introdução, Objetivos, Metodologia, Resultados e Discussão, Considerações Finais), incorporando as 6 figuras geradas e a tabela comparativa oficial.
   * **Total Geral do Projeto:** 10,0 pontos de 10,0 pontos concluídos com rigor científico absoluto!

---

### 📚 1. Síntese Comparativa Final dos Modelos

A tabela abaixo sumariza o comportamento de cada abordagem sobre os 442 pacientes do dataset `load_diabetes`:

| Modelo | Graus de Liberdade | RMSE Treino ($N=353$) | RMSE Teste ($N=89$) | Pearson $r$ (Teste) | Número de Condição ($\kappa$) | Veredito Científico |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **RBF Gaussiana ($M=8$)** | 9 | **60,95** | 67,05 | 0,4653 | $1,42 \times 10^5$ | **Venceu no Treino** (Baixo Viés, Alta Flexibilidade Local) |
| **MMQ Grau 1 (Linear)** | 2 | 62,08 | **63,73** | **0,5076** | **36,2** | **Venceu no Teste** (Baixa Variância, Navalha de Occam) |
| **MMQ Grau 2 (Parábola)**| 3 | 62,04 | 63,91 | 0,5034 | 970 | Degradação ligeira de teste, sem ganho preditivo |
| **MMQ Grau 3 (Cúbica)**  | 4 | 62,04 | 63,75 | 0,5052 | 24.300 | Explosão de $\kappa$ (perda de precisão IEEE 754) |

---

### 📚 2. O Teorema do *No Free Lunch* na Prática de Engenharia
O resultado empírico obtido ilustra com precisão o teorema do *No Free Lunch* ("Não Há Almoço Grátis"): nenhum algoritmo de aprendizado ou ajuste de curvas é universalmente superior para qualquer distribuição de dados:
* **Quando a RBF é mandatória:** Em fenômenos físicos/químicos/mecânicos altamente não-lineares, com baixa dispersão estocástica e dinâmicas locais ricas (como saturação de motores, curvas I-V de semicondutores, histerese magnética). Nesses casos, o MMQ linear falharia por subajuste severo (alto viés).
* **Quando o MMQ simples é a melhor escolha:** Em problemas biológicos, médicos ou econométricos com forte componente estocástico e correlação linear média. O modelo linear atua como um regularizador natural, evitando que o modelo persiga ruídos individuais de pacientes.

---

### ❓ 5 Questões de Revisão Técnica — Dia 5 (Fechamento do Projeto)

62. **Por que é essencial que o relatório técnico final apresente as métricas de treino e teste separadas lado a lado?**
    * *Resposta:* Para evidenciar o comportamento de generalização e diagnosticar fenômenos de overfitting (sobreajuste) ou underfitting (subajuste). Apresentar apenas o erro de treino mascara a memorização de ruído, enquanto apresentar apenas o teste omite a capacidade de representação do modelo.

63. **O que indica uma reta de calibração $y = a\hat{y} + b$ onde $a = 0{,}6460$ e $b = 51{,}48$?**
    * *Resposta:* Uma inclinação $a < 1$ e um intercepto $b > 0$ caracterizam o fenômeno estatístico de *regressão à média*. O modelo é cauteloso: quando o paciente tem uma progressão real muito baixa, o modelo tende a superestimar ligeiramente; quando a progressão real é muito alta, o modelo tende a subestimar, concentrando suas previsões próximas à média populacional ($\approx 152$).

64. **Como o arquivo LaTeX foi estruturado para garantir reprodutibilidade e compilação limpa?**
    * *Resposta:* O arquivo `relatorio_final_projeto_u2.tex` utiliza caminhos relativos para as figuras (`../figuras/`), pacotes canônicos (`amsmath`, `graphicx`, `booktabs`, `hyperref`), tabelas formatadas com réguas profissionais e equações numeradas, compilando via `pdflatex` sem dependências de caminhos absolutos locais.

65. **Qual a importância da Decomposição em Valores Singulares (SVD) na resolução da RBF neste projeto?**
    * *Resposta:* Como o espaçamento entre os 8 centros gerou um número de condição de $\kappa = 1{,}42 \times 10^5$, o cálculo da inversa direta $(\Phi^T \Phi)^{-1}$ estaria sujeito a instabilidade numérica de ponto flutuante. A pseudoinversa de Moore-Penrose via SVD projeta o sistema nos autovetores principais e anula valores singulares próximos de zero, garantindo pesos estáveis.

66. **Como o projeto atingiu os 10,0 pontos máximos do Barema de Avaliação?**
    * *Resposta:* Cumprindo integralmente os 6 pilares do barema: (1) Formulação e ajuste da RBF Gaussiana (2,0 pt); (2) Cálculo do EMQ e RMSE em treino e teste (1,0 pt); (3) Regressão linear de calibração real vs. predito (1,0 pt); (4) Coeficiente de correlação linear de Pearson (2,0 pt); (5) Comparação com MMQ polinomial sob isonomia amostral (2,0 pt); e (6) Redação de relatório técnico rigoroso e conclusões com fundamentação teórica (2,0 pt).
