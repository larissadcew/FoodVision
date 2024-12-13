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

### Equação 1:
![image](https://github.com/user-attachments/assets/3eed1cb4-5019-4ad5-924b-3ed607da3b2a)

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



















