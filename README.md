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

# Entendendo os Hiperparâmetros:
![image](https://github.com/user-attachments/assets/3b4624c3-cf43-447f-9901-1200058426cd)

# Comparando as Variantes ViT

![image](https://github.com/user-attachments/assets/3fa40379-eb07-4fa3-bc30-731e51da92df)

 Ao movermos de ViT-Base para ViT-Enorme, observamos um aumento gradual em todos os hiperparâmetros. Isso indica que o modelo está sendo escalado para lidar com tarefas mais complexas e conjuntos de dados maiores.Aumentar o número de camadas, o tamanho oculto e o número de cabeças geralmente leva a um melhor desempenho, mas também aumenta o custo computacional e o risco de overfitting. É importante encontrar um equilíbrio entre esses fatores.O ViT-Base é um bom ponto de partida para experimentos, pois oferece um bom equilíbrio entre desempenho e complexidade.

# Equação 1: Divida os dados em patches e crie a classe, posição e incorporação de patch.

Na forma vetorial, a incorporação pode ser algo como:

x_input = [class_token, image_patch_1, image_patch_2, image_patch_3...] + [class_token_position, image_patch_1_position, image_patch_2_position, image_patch_3_position...]

vamos começar criando as incorporações de classe, posição e patch para a arquitetura ViT.Começaremos com a incorporação do patch.Isso significa que transformaremos nossas imagens de entrada em uma sequência de patches e, em seguida, incorporaremos esses patches.Lembre-se de que uma incorporação é uma representação que pode ser aprendida de alguma forma e geralmente é um vetor.O termo aprendível é importante porque significa que a representação numérica de uma imagem de entrada (que o modelo vê) pode ser melhorada ao longo do tempo.Começaremos seguindo o parágrafo de abertura da seção 3.1 do artigo ViT (negrito meu):

O Transformer padrão recebe como entrada uma sequência 1D de incorporações de tokens. Para lidar com imagens 2D, remodelamos a imagem $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ em uma sequência de patches 2D achatados $\mathbf{x}_{p} \in \mathbb{R}^{N \times\left(P^{2} \cdot C\right)}$, onde $(H, W)$ é a resolução da imagem original, $C$ é o número de canais, $(P, P)$ é a resolução de cada patch de imagem e $N=H W / P^{2}$ é o número resultante de patches, que também serve como o comprimento efetivo da sequência de entrada para o Transformer. O Transformer usa tamanho de vetor latente constante $D$ em todas as suas camadas, então nivelamos os patches e mapeamos para dimensões de $D$ com uma projeção linear treinável (Eq. 1). Referimo-nos à saída dessa projeção como as incorporações de patch.

E tamanho estamos lidando com formas de imagem, vamos ter em mente a linha da Tabela 3 do artigo ViT:


































