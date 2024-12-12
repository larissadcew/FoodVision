# Análise e Implementação de Transformadores Visuais: Uma Abordagem Moderna para Reconhecimento de Imagens Baseada em Atenção

Neste blog, analisamos o trabalho pioneiro "Uma imagem vale 16x16 palavras: transformadores para reconhecimento de imagem em escala" publicado na ICLR, que revolucionou a abordagem do processamento de imagens através da adaptação da arquitetura de transformadores para visão computacional ViT adapta a arquitetura de transformadores - originalmente projetada para processamento de texto introduzida no artigo de pesquisa de aprendizado de máquina Atenção é tudo o que você precisa.(Destacamos a importância da modelagem baseada em atenção na revolução do processamento visual).  Reproduzimos a implementação desta metodologia, Este blog tem como objetivo fornecer aos pesquisadores e profissionais (1) uma compreensão aprofundada de como os transformadores podem ser efetivamente adaptados para tarefas de visão computacional, eliminando a necessidade de arquiteturas convolucionais complexas, e (2) uma análise detalhada de como a atenção visual baseada em patches pode ser implementada para criar modelos de reconhecimento de imagem mais eficientes e escaláveis.

![image](https://github.com/user-attachments/assets/f87cb646-8a58-4dd5-b814-b1eee75b0ddc)

# Introdução ao ViT 

Uma arquitetura moderna de aprendizado profundo geralmente é uma coleção de camadas e blocos. Onde as camadas pegam uma entrada (dados como uma representação numérica) e a manipulam usando algum tipo de função (por exemplo, a fórmula de autoatenção mostrada logo abaixo, no entanto, essa função pode ser quase qualquer coisa) e, em seguida, a produzem. Os blocos geralmente são pilhas de camadas umas sobre as outras, fazendo algo semelhante a uma única camada, mas várias vezes.[1]

![image](https://github.com/user-attachments/assets/18ae4852-74da-47c8-bd8e-b734d05cba5d)


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

  # Explorando as Quatro Equações:
  ![image](https://github.com/user-attachments/assets/a83ec89e-98be-4953-bd12-f38fe06e7352)

  Essas quatro equações representam a matemática por trás das quatro partes principais da arquitetura ViT.

A seção 3.1 descreve cada um deles (parte do texto foi omitido por brevidade, o texto em negrito é meu):


(1) : O Transformer usa tamanho de vetor latente constante $D$ em todas as suas camadas, então achatamos os patches e mapeamos para dimensões $D$ com uma projeção linear treinável (Eq. 1). Referimo-nos à saída desta projeção como as incorporações de patch... As incorporações de posição são adicionadas às incorporações de patch para reter informações posicionais. Usamos incorporações de posição 1D padrão que podem ser aprendidas..

(2) :O codificador Transformer (Vaswani et al., 2017) consiste em camadas alternadas de autoatenção de várias cabeças (MSA, consulte o Apêndice A) e blocos MLP (Eq. 2, 3). Layernorm (LN) é aplicado antes de cada bloco e conexões residuais após cada bloco (Wang et al., 2019; Baevski & Auli, 2019).

(3) :O mesmo que a equação 2.

(4):Semelhante ao token [ class ] do BERT, precedemos uma incorporação que pode ser aprendida na sequência de patches incorporados $\left(\mathbf{z}_{0}^{0}=\mathbf{x}_{\text {class }}\right)$, cujo estado na saída do codificador Transformer $\left(\mathbf{z}_{L}^{0}\right)$ serve como a representação da imagem $\mathbf{y}$ (Eq. 4)..

Vamos mostrrar como isso se apica :
![image](https://github.com/user-attachments/assets/f5242bc7-8e81-4e4e-9175-2c76fb421f0f)
Conectando a Figura 1 do artigo ViT às quatro equações da seção 3.1 que descrevem a matemática por trás de cada uma das camadas / blocos.
Há muita coisa acontecendo na imagem acima, mas seguir as linhas coloridas e setas revela os principais conceitos da arquitetura ViT.

# DIVIDINDO as 4 equaçoes ainda MAIS:

# Visão geral da equação 1:

Esta equação lida com o token de classe, a incorporação de patch e a incorporação de posição ($\mathbf{E}$ é para incorporação) da imagem de entrada.

Na forma vetorial, a incorporação pode ser algo como

x_input = [class_token, image_patch_1, image_patch_2, image_patch_3...] + [class_token_position, image_patch_1_position, image_patch_2_position, image_patch_3_position...]

# Visão geral da equação 2:

Isso diz que para cada camada de $1$ a $L$ (o número total de camadas), há uma camada de atenção de várias cabeças (MSA) envolvendo uma camada LayerNorm (LN).

A adição no final é o equivalente a adicionar a entrada à saída e formar uma conexão de salto / residual.

No pseudocódigo, isso pode ser semelhante a:
x_output_MSA_block = MSA_layer(LN_layer(x_input)) + x_input

Observe a conexão de salto no final (adicionando a entrada das camadas à saída das camadas).

# Visão geral da equação 3:

Isso diz que para cada camada de $1$ a $L$ (o número total de camadas), há também uma camada Multilayer Perceptron (MLP) envolvendo uma camada LayerNorm (LN).

A adição no final mostra a presença de uma conexão de salto / residual.

Chamaremos essa camada de "bloco MLP".

No pseudocódigo, isso pode ser semelhante a:

x_output_MLP_block = MLP_layer(LN_layer(x_output_MSA_block)) + x_output_MSA_block

# Visão geral da equação 4:

Isso diz que para a última camada $L$, a saída $y$ é o token de índice 0 de $z$ encapsulado em uma camada LayerNorm (LN).

Ou, no nosso caso, o índice 0 de :x_output_MLP_block

y = Linear_layer(LN_layer(x_output_MLP_block[0]))

É claro que existem algumas simplificações acima, mas cuidaremos delas quando começarmos a escrever o código PyTorch para cada seção.


# Explorando a Tabela 1(None)

# Equação 1: Divida os dados em patches e crie a classe, posição e incorporação de patch.


Começaremos com a incorporação do patch.Isso significa que transformaremos nossas imagens de entrada em uma sequência de patches e, em seguida, incorporaremos esses patches.
Lembre-se de que uma incorporação é uma representação que pode ser aprendida de alguma forma e geralmente é um vetor.O termo aprendível é importante porque significa que a representação numérica de uma imagem de entrada (que o modelo vê) pode ser melhorada ao longo do tempo.

Essa técnica permite que um modelo Transformer, originalmente projetado para processar sequências 1D (como texto), trabalhe com dados 2D (imagens) ao converter a imagem em uma sequência de vetores que representam partes menores da imagem.

Começaremos seguindo o parágrafo de abertura da seção 3.1 do artigo ViT (negrito meu):

```
O Transformer padrão recebe como entrada uma sequência 1D de incorporações de tokens. Para lidar com imagens 2D, remodelamos a imagem $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ em uma sequência de patches 2D achatados $\mathbf{x}_{p} \in \mathbb{R}^{N \times\left(P^{2} \cdot C\right)}$, onde $(H, W)$ é a resolução da imagem original, $C$ é o número de canais, $(P, P)$ é a resolução de cada patch de imagem e $N=H W / P^{2}$ é o número resultante de patches, que também serve como o comprimento efetivo da sequência de entrada para o Transformer. O Transformer usa tamanho de vetor latente constante $D$ em todas as suas camadas, então nivelamos os patches e mapeamos para dimensões de $D$ com uma projeção linear treinável (Eq. 1). Referimo-nos à saída dessa projeção como as incorporações de patch
```
E tamanho estamos lidando com formas de imagem, vamos ter em mente a linha da Tabela 3 do artigo ViT:
```
A resolução do treinamento é 224.

```
Detalhando todas as partes:

Você começa com uma imagem representada por x ∈ RH×W×C,onde:
- H é a altura da imagem em pixels.
- W é a largura da imagem em pixels.
- C é o número de canais de cor (ex: 3 para RGB)
  
### Dividindo em patches:

Ao invés de tratar a imagem inteira como uma única entrada, ela é dividida em pequenos pedaços quadrados chamados patches. Cada patch tem resolução.

### Criando a sequência:
Esses patches 2D são então "achatados" em vetores 1D. Imagine pegar cada quadrado e esticar seus pixels em uma linha. Isso resulta em xp ∈ Rn×(P2⋅C),onde:

- N = HW/P2  é o número total de patches. Como a imagem original tem H×W pixels e cada patch tem P×P pixels, o número de patches é a divisão da área total pela área de cada patch.
- P2⋅C  o tamanho de cada patch achatado. Cada patch tem P2 pixels, e cada pixel tem C canais de cor, então o tamanho do vetor resultante é o produto desses dois.

### Exemplo:
Imagine uma imagem de 224x224 pixels com 3 canais (RGB). Se você escolher patches de 16x16 pixels, você terá (224*224)/(16*16) = 196 patches. Cada patch achatado terá 16*16*3 = 768 elementos.

![image](https://github.com/user-attachments/assets/f17b99f5-3a1d-485e-a837-c5e1ac288b17)

### Incorporação (Embedding) dos patches

### Tamanho do vetor latente D:
O Transformer trabalha com vetores de um tamanho fixo, chamado de tamanho do vetor latente, representado por D. Este valor é constante em todas as camadas do Transformer.

### Projeção Linear:

Os patches achatados, que têm tamanho P2⋅C, precisam ser transformados para esse tamanho D. Isso é feito através de uma projeção linear treinável. Em termos matemáticos, isso pode ser representado como:
- ​Embedding =p⋅W+b

  onde:
- W é uma matriz de pesos de tamanho (P2⋅C)×D .Esta matriz é treinada durante o aprendizado do modelo.
- b é um vetor de bias (viés) de tamanho D.
 Incorporações de patch: O resultado dessa projeção é chamado de "incorporação de patch". Cada patch agora é representado por um vetor de tamanho D, adequado para entrada no Transformer.

Juntando tudo:

```
# Setup hyperparameters and make sure img_size and patch_size are compatible
img_size = 224
patch_size = 16
num_patches = img_size/patch_size
assert img_size % patch_size == 0, "Image size must be divisible by patch size"
print(f"Number of patches per row: {num_patches}\
        \nNumber of patches per column: {num_patches}\
        \nTotal patches: {num_patches*num_patches}\
        \nPatch size: {patch_size} pixels x {patch_size} pixels")

# Create a series of subplots
fig, axs = plt.subplots(nrows=img_size // patch_size, # need int not float
                        ncols=img_size // patch_size,
                        figsize=(num_patches, num_patches),
                        sharex=True,
                        sharey=True)

# Loop through height and width of image
for i, patch_height in enumerate(range(0, img_size, patch_size)): # iterate through height
    for j, patch_width in enumerate(range(0, img_size, patch_size)): # iterate through width

        # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
        axs[i, j].imshow(image_permuted[patch_height:patch_height+patch_size, # iterate through height
                                        patch_width:patch_width+patch_size, # iterate through width
                                        :]) # get all color channels

        # Set up label information, remove the ticks for clarity and set labels to outside
        axs[i, j].set_ylabel(i+1,
                             rotation="horizontal",
                             horizontalalignment="right",
                             verticalalignment="center")
        axs[i, j].set_xlabel(j+1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].label_outer()

# Set a super title
fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
plt.show()
```




























