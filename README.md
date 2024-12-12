# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
Neste blog, mergulhamos no artigo ICLR 2019 Reward Constrained Policy Optimization (RCPO) de Tessler et al. e destacamos a importância da modelagem adaptativa de recompensas no aprendizado por reforço seguro. Reproduzimos os resultados experimentais do artigo implementando o RCPO na Otimização de Políticas Proximais (PPO). Este blog tem como objetivo fornecer aos pesquisadores e profissionais (1) uma melhor compreensão do aprendizado por reforço seguro em termos de otimização restrita e (2) como as funções de recompensa penalizadas podem ser efetivamente usadas para treinar uma política robusta https://arxiv.org/pdf/2010.11929
![image](https://github.com/user-attachments/assets/f87cb646-8a58-4dd5-b814-b1eee75b0ddc)



# abc
Mais especificamente, vamos replicar o artigo de pesquisa de aprendizado de máquina Uma imagem vale 16x16 palavras: transformadores para reconhecimento de imagem em escala (artigo ViT) com PyTorch.
A arquitetura de rede neural do Transformer foi originalmente introduzida no artigo de pesquisa de aprendizado de máquina Atenção é tudo o que você precisa.
E a arquitetura original do Transformer foi projetada para funcionar em sequências unidimensionais (1D) de texto.
Uma arquitetura Transformer é geralmente considerada qualquer rede neural que usa o mecanismo de atenção) como sua camada de aprendizado primária. Semelhante a como uma rede neural convolucional (CNN) usa convoluções como sua principal camada de aprendizado.
Como o nome sugere, a arquitetura Vision Transformer (ViT) foi projetada para adaptar a arquitetura original do Transformer ao(s) problema(s) de visão (classificação sendo a primeira e desde então muitas outras se seguiram).

