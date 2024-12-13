# Análise e Implementação de Transformadores Visuais: Uma Abordagem Moderna para Reconhecimento de Imagens Baseada em Atenção

Neste blog, analisamos o trabalho pioneiro "Uma imagem vale 16x16 palavras: transformadores para reconhecimento de imagem em escala" publicado na ICLR, que revolucionou a abordagem do processamento de imagens através da adaptação da arquitetura de transformadores para visão computacional ViT adapta a arquitetura de transformadores - originalmente projetada para processamento de texto introduzida no artigo de pesquisa de aprendizado de máquina Atenção é tudo o que você precisa.(Destacamos a importância da modelagem baseada em atenção na revolução do processamento visual).  Reproduzimos a implementação desta metodologia, Este blog tem como objetivo fornecer aos pesquisadores e profissionais (1) uma compreensão aprofundada de como os transformadores podem ser efetivamente adaptados para tarefas de visão computacional, eliminando a necessidade de arquiteturas convolucionais complexas, e (2) uma análise detalhada de como a atenção visual baseada em patches pode ser implementada para criar modelos de reconhecimento de imagem mais eficientes e escaláveis.

![image](https://github.com/user-attachments/assets/f87cb646-8a58-4dd5-b814-b1eee75b0ddc)

# Introdução ao ViT 

Uma arquitetura moderna de aprendizado profundo geralmente é uma coleção de camadas e blocos. Onde as camadas pegam uma entrada (dados como uma representação numérica) e a manipulam usando algum tipo de função (por exemplo, a fórmula de autoatenção mostrada logo abaixo, no entanto, essa função pode ser quase qualquer coisa) e, em seguida, a produzem. Os blocos geralmente são pilhas de camadas umas sobre as outras, fazendo algo semelhante a uma única camada, mas várias vezes.[1]

![image](https://github.com/user-attachments/assets/18ae4852-74da-47c8-bd8e-b734d05cba5d)

Vamos começar examinando a Figura 1 do ViT Paper

Explorando a Figura 1
As principais coisas às quais prestaremos atenção são:

- Camadas - recebe uma entrada, executa uma operação ou função na entrada, produz uma saída
- Blocos  - uma coleção de camadas, que por sua vez também recebe uma entrada e produz uma saída.
  
![image](https://github.com/user-attachments/assets/5223448b-4fe6-4b75-b940-e2367458dfbb)

# visao geral da arquitetura vit: 

- Patch + Incorporação de Posição (entradas) - Transforma a imagem de entrada em uma sequência de patches de imagem e adiciona um número de posição para especificar em que ordem o patch vem.
- Projeção linear de patches achatados (Patches Incorporados) - Os patches de imagem são transformados em uma incorporação, o benefício de usar uma incorporação em vez de apenas os valores da imagem é que uma incorporação é uma representação que pode ser aprendida (normalmente na forma de um vetor) da imagem que pode melhorar com o treinamento.
- Norm - É a abreviação de "Layer Normalization" ou "LayerNorm", uma técnica para regularizar (reduzir o sobreajuste) de uma rede neural, você pode usar LayerNorm por meio da camada PyTorch torch.nn.LayerNorm().
- Atenção de várias cabeças - Esta é uma camada de autoatenção de várias cabeças ou "MSA" para abreviar. Você pode criar uma camada MSA por meio da camada PyTorch torch.nn.MultiheadAttention().
- MLP (ou perceptron multicamada) - Um MLP geralmente pode se referir a qualquer coleção de camadas feedforward (ou, no caso do PyTorch, uma coleção de camadas com um método). No ViT Paper, os autores se referem ao MLP como "bloco MLP" e contém duas camadas torch.nn.Linear() com uma ativação de não linearidade torch.nn.GELU() entre elas (seção 3.1) e uma camada torch.nn.Dropout() após cada uma (Apêndice B.1).forward()
- Transformer Encoder - O Transformer Encoder é uma coleção das camadas listadas acima. Existem duas conexões de salto dentro do codificador Transformer (os símbolos "+"), o que significa que as entradas da camada são alimentadas diretamente para as camadas imediatas, bem como para as camadas subsequentes. A arquitetura ViT geral é composta por vários codificadores Transformer empilhados uns sobre os outros.
- MLP Head - Esta é a camada de saída da arquitetura, ela converte os recursos aprendidos de uma entrada em uma saída de classe. Como estamos trabalhando na classificação de imagens, você também pode chamar isso de "cabeça do classificador". A estrutura do MLP Head é semelhante ao bloco MLP.

Você pode notar que muitas das partes da arquitetura ViT podem ser criadas com camadas PyTorch existentes 

  # Explorando as Quatro Equações:
  ![image](https://github.com/user-attachments/assets/a83ec89e-98be-4953-bd12-f38fe06e7352)

  Essas quatro equações representam a matemática por trás das quatro partes principais da arquitetura ViT.

A seção 3.1 descreve cada um deles (parte do texto foi omitido por brevidade, o texto em negrito é meu):

(1) : O Transformer usa tamanho de vetor latente constante $D$ em todas as suas camadas, então achatamos os patches e mapeamos para dimensões $D$ com uma projeção linear treinável (Eq. 1). Referimo-nos à saída desta projeção como as incorporações de patch... As incorporações de posição são adicionadas às incorporações de patch para reter informações posicionais. Usamos incorporações de posição 1D padrão que podem ser aprendidas..

(2) :O codificador Transformer (Vaswani et al., 2017) consiste em camadas alternadas de autoatenção de várias cabeças (MSA, consulte o Apêndice A) e blocos MLP (Eq. 2, 3). Layernorm (LN) é aplicado antes de cada bloco e conexões residuais após cada bloco (Wang et al., 2019; Baevski & Auli, 2019).

(3) :O mesmo que a equação 2.

(4):Semelhante ao token [ class ] do BERT, precedemos uma incorporação que pode ser aprendida na sequência de patches incorporados $\left(\mathbf{z}_{0}^{0}=\mathbf{x}_{\text {class }}\right)$, cujo estado na saída do codificador Transformer $\left(\mathbf{z}_{L}^{0}\right)$ serve como a representação da imagem $\mathbf{y}$ (Eq. 4)..

### Layernorm (LN) é aplicado antes de cada bloco e conexões residuais após cada bloco. O MLP contém duas camadas com uma não linearidade GELU.
![image](https://github.com/user-attachments/assets/1ac0382a-b603-4835-8a7e-437c3ecf57f4)

Os modelos "Base" e "Large" são adotados diretamente do BERT e o modelo "Huge" maior é adicionado
ViT-L/16 significa a variante "Grande" com tamanho de patch de entrada 16×16. Observe que o comprimento da sequência do Transformer é inversamente proporcional ao quadrado do tamanho do patch, e os modelos com tamanho de patch menor são computacionalmente mais caros.

### Entrando fundo em cada equaçao:

# Equação 1:

### A fórmula que mostramos, x ∈ ℝ^(H×W×C) -> x_p ∈ ℝ^(N×(P²⋅C)), descreve matematicamente essa transformação inicial da imagem em uma sequência de embeddings.

### x: A imagem original.
### H, W, C: Altura, largura e número de canais da imagem, respectivamente.
### x_p: A imagem após ser dividida em patches e transformada em embeddings.
### N: Número total de patches.
### P: Tamanho de cada patch.
### C: Número de canais da imagem.

![image](https://github.com/user-attachments/assets/3eed1cb4-5019-4ad5-924b-3ed607da3b2a)

### Equação 1.1 Transformar uma imagem em patche.
Transformar uma imagem em patches é o primeiro passo crucial para que o ViT possa processar e entender imagens. Essa técnica permite que o modelo capture tanto informações locais quanto globais da imagem, tornando-o uma ferramenta poderosa para diversas tarefas de visão computacional.
Imagine que você está tentando descrever uma imagem para alguém. Você não descreveria a imagem pixel por pixel. Em vez disso, você dividiria a imagem em partes (objetos, pessoas, paisagens) e descreveria cada parte individualmente, bem como as relações entre elas. É assim que o ViT funciona: ele divide a imagem em patches (as "palavras" da imagem) e aprende a "linguagem" da visão, entendendo as relações entre esses patches.

[imagem de peths]

### Equação 1.2 Criando patches de imagem com Pytorch

Os autores do artigo do Vision Transformer (ViT) mencionam na seção 3.1 que a incorporação de patches pode ser alcançada usando uma rede neural convolucional (CNN):

Arquitetura híbrida. Como alternativa aos patches de imagem brutos, a sequência de entrada pode ser formada a partir de mapas de recursos de uma CNN (LeCun et al., 1989). Neste modelo híbrido, a projeção de incorporação de patch $\mathbf{E}$ (Eq. 1) é aplicada a patches extraídos de um mapa de recursos da CNN. Como um caso especial, os patches podem ter tamanho espacial $1 \times 1$, o que significa que a sequência de entrada é obtida simplesmente nivelando as dimensões espaciais do mapa de recursos e projetando para a dimensão Transformer. A incorporação de entrada de classificação e as incorporações de posição são adicionadas conforme descrito acima.

O "mapa de recursos" a que eles se referem são os pesos / ativações produzidos por uma camada convolucional que passa sobre uma determinada imagem.

Ao definir os parâmetros de kernel_size e passada de uma camada torch.nn.Conv2d() igual ao patch_size, podemos efetivamente obter uma camada que divide nossa imagem em patches e cria uma incorporação que pode ser aprendida (chamada de "Projeção Linear" no artigo ViT) de cada patch.

Podemos recriá-los com torch.nn.Conv2d() por transformar nossa imagem em patches de mapas de recursos da CNN ou torch.nn.Flatten() para nivelar as dimensões espaciais do mapa de recursos.

### Equação 1.3 Achatando o patch incorporado:






![image](https://github.com/user-attachments/assets/17e8dfe0-8d5e-4b68-97e6-7a54c4994c66)

<img alt="Animação da arquitetura do transformador de visão, passando de uma única imagem e passando-a por uma camada de patch embedidng e, em seguida, passando-a pelo codificador do transformador." src="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/images/08-vit-paper-architecture-animation-full-architecture.gif" width="900/" _mstalt="15581254" _msthash="832">

Animando todo o fluxo de trabalho ViT: desde a incorporação de patches até o codificador do transformador e o cabeçote MLP.
Do ponto de vista do código, criar a incorporação de patches é provavelmente a maior seção de replicação do artigo ViT.
Muitas das outras partes do papel ViT, como as camadas Multi-Head Attention e Norm, podem ser criadas usando camadas PyTorch existentes.

### Equação 2:
![image](https://github.com/user-attachments/assets/f3b5e4cb-ba69-44fd-8ab8-04277bf9829b)

### Equação 3:
![image](https://github.com/user-attachments/assets/669d563b-6335-417f-b7cb-00a2ffe7c5b8)


### Equação 4:
![image](https://github.com/user-attachments/assets/c3757e04-a589-4973-ab23-2145c2757bdd)


### Crie o codificador do transformador:

### Juntando tudo para criar o ViT
Food Vision


### Criando um otimizador:

Pesquisando o artigo ViT por "otimizador", a seção 4.1 sobre Treinamento e Ajuste Fino afirma:
Treinamento & Ajuste Fino. Treinamos todos os modelos, incluindo ResNets, usando Adam (Kingma & Ba, 2015) com $\beta_{1}=0,9, \beta_{2}=0,999$, um tamanho de lote de 4096 e aplicamos um decaimento de peso alto de $0,1$, que descobrimos ser útil para a transferência de todos os modelos (o Apêndice D.1 mostra que, em contraste com as práticas comuns, Adam funciona um pouco melhor do que o SGD para ResNets em nosso ambiente).

Portanto, podemos ver que eles escolheram usar o otimizador "Adam" (torch.optim.Adam()) em vez de SGD (gradiente descendente estocástico, torch.optim.SGD()).
Criando uma função de perda:

### Treinando nosso modelo ViT:

Ok, agora que sabemos qual otimizador e função de perda vamos usar, vamos configurar o código de treinamento para treinar nosso ViT.
Começaremos importando o script e, em seguida, configuraremos o otimizador e a função de perda e, finalmente, usaremos a função de para treinar nosso modelo ViT por 10 épocas (estamos usando um número menor de épocas do que o artigo ViT para garantir que tudo funcione).

### Plotar as curvas de perda do nosso modelo ViT:

![image](https://github.com/user-attachments/assets/0470e29e-8c80-4765-af7d-2b3c7ab160d3)

Pelo menos a perda parece estar indo na direção certa, mas as curvas de precisão não são muito promissoras.

Esses resultados provavelmente se devem à diferença nos recursos de dados e no regime de treinamento de nosso modelo ViT em relação ao artigo ViT
Que tal vermos se podemos consertar isso trazendo um modelo ViT pré-treinado?

### Porque um modelo pretreinado:

Embora nossa arquitetura ViT seja a mesma do artigo, os resultados do artigo ViT foram alcançados usando muito mais dados e um esquema de treinamento mais elaborado do que o nosso.
Devido ao tamanho da arquitetura ViT e seu alto número de parâmetros (maior capacidade de aprendizado) e quantidade de dados que ela usa (aumento das oportunidades de aprendizado), muitas das técnicas usadas no esquema de treinamento em papel ViT, como aquecimento da taxa de aprendizado, decaimento da taxa de aprendizado e recorte de gradiente, são projetadas especificamente para evitar o sobreajuste (regularização).
A boa notícia é que existem muitos modelos ViT pré-treinados (usando grandes quantidades de dados) disponíveis online.

## usando um modelo preteinado:



















