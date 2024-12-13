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
- 

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
















