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

### ❓ 10 Questões de Revisão Técnica — Dia 2 (Preparação para Defesa Oral)

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
