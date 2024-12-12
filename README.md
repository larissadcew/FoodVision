# Análise e Implementação de Transformadores Visuais: Uma Abordagem Moderna para Reconhecimento de Imagens Baseada em Atenção

Neste blog, analisamos o trabalho pioneiro "Uma imagem vale 16x16 palavras: transformadores para reconhecimento de imagem em escala" publicado na ICLR, que revolucionou a abordagem do processamento de imagens através da adaptação da arquitetura de transformadores para visão computacional ViT adapta a arquitetura de transformadores - originalmente projetada para processamento de texto introduzida no artigo de pesquisa de aprendizado de máquina Atenção é tudo o que você precisa.(Destacamos a importância da modelagem baseada em atenção na revolução do processamento visual).  Reproduzimos a implementação desta metodologia, Este blog tem como objetivo fornecer aos pesquisadores e profissionais (1) uma compreensão aprofundada de como os transformadores podem ser efetivamente adaptados para tarefas de visão computacional, eliminando a necessidade de arquiteturas convolucionais complexas, e (2) uma análise detalhada de como a atenção visual baseada em patches pode ser implementada para criar modelos de reconhecimento de imagem mais eficientes e escaláveis.

![image](https://github.com/user-attachments/assets/f87cb646-8a58-4dd5-b814-b1eee75b0ddc)

# Introdução ao ViT 

Uma arquitetura moderna de aprendizado profundo geralmente é uma coleção de camadas e blocos. Onde as camadas pegam uma entrada (dados como uma representação numérica) e a manipulam usando algum tipo de função (por exemplo, a fórmula de autoatenção mostrada logo abaixo, no entanto, essa função pode ser quase qualquer coisa) e, em seguida, a produzem. Os blocos geralmente são pilhas de camadas umas sobre as outras, fazendo algo semelhante a uma única camada, mas várias vezes.[1]

![image](https://github.com/user-attachments/assets/18ae4852-74da-47c8-bd8e-b734d05cba5d)

Vamos começar examinando a Figura 1 do ViT Paper
As principais coisas às quais prestaremos atenção são:

- Camadas - recebe uma entrada, executa uma operação ou função na entrada, produz uma saída.
- Blocos - uma coleção de camadas, que por sua vez também recebe uma entrada e produz uma saída.
  ![image](https://github.com/user-attachments/assets/5223448b-4fe6-4b75-b940-e2367458dfbb)

  A arquitetura ViT é composta por vários estágios:

- Patch + Incorporação de Posição (entradas) - Transforma a imagem de entrada em uma sequência de patches de imagem e adiciona um número de posição para especificar em que ordem o patch vem.
- Projeção linear de patches achatados (Patches Incorporados) - Os patches de imagem são transformados em uma incorporação, o benefício de usar uma incorporação em vez de apenas os valores da imagem é que uma incorporação é uma representação que pode ser aprendida (normalmente na forma de um vetor) da imagem que pode melhorar com o treinamento.
- Norm - É a abreviação de "Layer Normalization" ou "LayerNorm", uma técnica para regularizar (reduzir o sobreajuste) de uma rede neural, você pode usar LayerNorm por meio da camada PyTorch torch.nn.LayerNorm().
- Atenção de várias cabeças - Esta é uma camada de autoatenção de várias cabeças ou "MSA" para abreviar. Você pode criar uma camada MSA por meio da camada PyTorch torch.nn.MultiheadAttention().
- MLP (ou perceptron multicamada) - Um MLP geralmente pode se referir a qualquer coleção de camadas feedforward (ou, no caso do PyTorch, uma coleção de camadas com um método). No ViT Paper, os autores se referem ao MLP como "bloco MLP" e contém duas camadas torch.nn.Linear() com uma ativação de não linearidade torch.nn.GELU() entre elas (seção 3.1) e uma camada torch.nn.Dropout() após cada uma (Apêndice B.1).forward()
- Transformer Encoder - O Transformer Encoder é uma coleção das camadas listadas acima. Existem duas conexões de salto dentro do codificador Transformer (os símbolos "+"), o que significa que as entradas da camada são alimentadas diretamente para as camadas imediatas, bem como para as camadas subsequentes. A arquitetura ViT geral é composta por vários codificadores Transformer empilhados uns sobre os outros.
- MLP Head - Esta é a camada de saída da arquitetura, ela converte os recursos aprendidos de uma entrada em uma saída de classe. Como estamos trabalhando na classificação de imagens, você também pode chamar isso de "cabeça do classificador". A estrutura do MLP Head é semelhante ao bloco MLP.
  
![image](https://github.com/user-attachments/assets/0130d970-645d-484a-b751-fff0ea761721)

Essas quatro equações representam a matemática por trás das quatro partes principais da arquitetura ViT.

A seção 3.1 descreve cada um deles (parte do texto foi omitido por brevidade, o texto em negrito é meu):
![image](https://github.com/user-attachments/assets/d7b112df-6a10-4092-8692-f35440436572)

![image](https://github.com/user-attachments/assets/46cde5dd-c1d3-4ec0-9dcd-696d784ce454)

Conectando a Figura 1 do artigo ViT às quatro equações da seção 3.1 que descrevem a matemática por trás de cada uma das camadas / blocos.

Há muita coisa acontecendo na imagem acima, mas seguir as linhas coloridas e setas revela os principais conceitos da arquitetura ViT.

Que tal dividirmos cada equação ainda mais (será nosso objetivo recriá-las com código)?

Em todas as equações (exceto a equação 4), "$\mathbf{z}$" é a saída bruta de uma camada específica.


Visão geral da equação 1

````
  z₀ = [x_class; x_p^1E; x_p^2E; ...; x_p^NE] + E_pos
````
- z₀: Representa a saída da camada inicial de incorporação de patch. É o vetor que contém a representação inicial da imagem.
- x_class: É o token de classe, um vetor que representa a classe da imagem inteira.
- x_p^i: É o i-ésimo patch da imagem, após ser transformado em um vetor.
- E: É a matriz de incorporação que mapeia os patches para um espaço de representação de dimensão D.
- E_pos: É a matriz de incorporação de posição, que adiciona informações sobre a posição espacial de cada patch.
  ![image](https://github.com/user-attachments/assets/d2b65ee4-a6ae-4931-9a7b-37c839a2a236)
  Onde cada um dos elementos no vetor pode ser aprendido (seu ).requires_grad=True

  Visão geral da equação 2(Bloco MSA na ViT)

  ```
  zₗ' = MSA(LN(zₗ₋₁)) + zₗ₋₁

  ```
  zₗ': Representa a saída do bloco MSA na camada l.
  MSA: É a função de atenção multi-cabeça. Ela calcula a importância relativa de diferentes partes da entrada.
  LN: É a normalização por camada (Layer Normalization), que ajuda a estabilizar o treinamento da rede neural.
  zₗ₋₁: É a saída da camada anterior.
  - Normalização: A entrada da camada (zₗ₋₁) passa por uma normalização por camada. Isso ajuda a manter os valores das ativações em uma faixa razoável, acelerando o treinamento e melhorando a estabilidade.
  -  Atenção Multi-Cabeça: A saída da normalização é então processada pela camada de atenção multi-cabeça. Essa camada permite que o modelo aprenda relações complexas entre diferentes partes da entrada. Cada     "cabeça" de atenção se concentra em diferentes aspectos da entrada, aumentando a capacidade do modelo de capturar informações relevantes.
  -  . Conexão Residual: A saída da camada de atenção é adicionada à entrada original (zₗ₋₁). Essa técnica, conhecida como conexão residual, ajuda a aliviar o problema do gradiente vanishing, permitindo que o modelo aprenda representações mais profundas.
 ![image](https://github.com/user-attachments/assets/d240f278-c0e8-46b6-9238-c5f90b8b6631)

  ```
   zₗ = MLP(LN(zₗ')) + zₗ'
  ```
- zₗ: Representa a saída final do bloco MLP na camada l. É a informação que sai desse bloco e será utilizada nas próximas camadas ou na saída final do modelo.
- MLP: É a camada Multilayer Perceptron, uma rede neural feedforward com múltiplas camadas. Ela é responsável por aplicar transformações não-lineares aos dados, aprendendo representações mais complexas.
- LN: É a camada de normalização por camada (Layer Normalization). Ela normaliza os valores de entrada para cada neurônio, ajudando a estabilizar o treinamento e acelerar a convergência.
- zₗ': É a saída do bloco MSA da camada l. Essa saída, que já contém informações sobre as relações entre os diferentes elementos da entrada, serve como entrada para o bloco MLP.

  ```
   y = LN(z_L^0)Mapeando a parte de incorporação de patch e posição da arquitetura ViT da Figura 1 à Equação 1. O parágrafo de abertura da seção 3.1 descreve as diferentes formas de entrada e saída da camada de incorporação de patch.
  ```
  - y: Representa a saída final do modelo. É o vetor que contém as informações sobre a classificação da imagem ou qualquer outra tarefa.
  - LN: É a camada de normalização por camada (Layer Norm). Ela normaliza os valores de entrada, ajudando a estabilizar o treinamento e melhorar o desempenho.
  - z_L^0: Representa o token de classe na última camada (L). O token de classe é um vetor especial que foi adicionado no início da sequência e contém informações sobre a classificação da imagem inteira.


1.incorporação do patch.

Isso significa que transformaremos nossas imagens de entrada em uma sequência de patches e, em seguida, incorporaremos esses patches.

Começaremos seguindo o parágrafo de abertura da seção 3.1 do artigo ViT:

"O Transformer padrão recebe como entrada uma sequência 1D de incorporações de tokens. Para lidar com imagens 2D, remodelamos a imagem x∈R H×W×C em uma sequência de patches 2D achatados x p ∈R N×(P 2⋅C),onde (H,W) é a resolução da imagem original, C é o número de canais (P,P) é a resolução de cada patch de imagem e N=HW/P 2 é o número resultante de patches, que também serve como o comprimento efetivo da sequência de entrada para o Transformer.O Transformer usa tamanho de vetor latente constante D em todas as suas camadas, então nivelamos os patches e mapeamos para dimensões de D com uma projeção linear treinável (Eq. 1).Referimo-nos à saída dessa projeção como as incorporações de patch."

Vamos decifrar isso:

- imagem de Entrada: Uma imagem com altura H largura W e C canais a W e C canais (ex: RGB tem 3 canais) é representada como  x∈ R H×W×C.

- Divisão em Patches: A imagem é dividida em N patches, cada um com resolução P×P.Cada patch tem tamanho P 2⋅C (pixels vezes o número de canais).A sequência de patches achatados é representada como xp ∈R N×(P 2⋅C).
​
- Número de Patches:O número total de patches N é calculado como  N = P 2 H×W

- Incorporação de Patch: Cada patch é então linearmente projetado para um espaço de dimensão D. Essa projeção é treinável, o que significa que os pesos dessa projeção são aprendidos durante o treinamento do modelo. O resultado dessa projeção é chamado de "incorporação de patch".


Incorporações de Classe e Posição:

Além das incorporações de patch, o ViT utiliza:

- Incorporação de Classe: Um token especial de "classe" é adicionado à sequência de incorporações de patch. A representação deste token após passar pelo Transformer serve como a representação da imagem inteira para tarefas de classificação.

- Incorporação de Posição: Como os Transformers não têm noção inerente de ordem sequencial, incorporações de posição são adicionadas às incorporações de patch para fornecer informações sobre a posição relativa dos patches na imagem original.

  Exemplo Prático:
  ### Imagine uma imagem de 224x224 pixels (como mencionado na Tabela 3 do artigo) e patches de 16x16 pixels:
  - H=224, W=224, P=16
  - N = 16×16 = 196 patches
       224×224
  Cada patch é então transformado em um vetor de tamanho D (o valor de D depende da configuração do modelo ViT, veja a Tabela 1 do artigo)





![image](https://github.com/user-attachments/assets/fba1bd66-31db-400e-9420-07899bfb4264)


  










<img alt="example of creating a patch embedding by passing a convolutional layer over a single image" src="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/images/08-vit-paper-patch-embedding-animation.gif" width="900" _msthidden="A" _mstalt="4690543" _msthash="3251">

























