# Análise e Implementação de Transformadores Visuais: Uma Abordagem Moderna para Reconhecimento de Imagens Baseada em Atenção

# resumo do modelo

https://arxiv.org/pdf/2010.11929
![image](https://github.com/user-attachments/assets/f87cb646-8a58-4dd5-b814-b1eee75b0ddc)

# Introdução ao o que e um vit
Mais especificamente, vamos replicar o artigo de pesquisa de aprendizado de máquina Uma imagem vale 16x16 palavras: transformadores para reconhecimento de imagem em escala (artigo ViT) com PyTorch.
A arquitetura de rede neural do Transformer foi originalmente introduzida no artigo de pesquisa de aprendizado de máquina Atenção é tudo o que você precisa.
E a arquitetura original do Transformer foi projetada para funcionar em sequências unidimensionais (1D) de texto.
Uma arquitetura Transformer é geralmente considerada qualquer rede neural que usa o mecanismo de atenção) como sua camada de aprendizado primária. Semelhante a como uma rede neural convolucional (CNN) usa convoluções como sua principal camada de aprendizado.
Como o nome sugere, a arquitetura Vision Transformer (ViT) foi projetada para adaptar a arquitetura original do Transformer ao(s) problema(s) de visão (classificação sendo a primeira e desde então muitas outras se seguiram).




![image](https://github.com/user-attachments/assets/6f79a9e6-44f6-407e-b809-e27817f5aaf0)


Uma arquitetura moderna de aprendizado profundo geralmente é uma coleção de camadas e blocos. Onde as camadas pegam uma entrada (dados como uma representação numérica) e a manipulam usando algum tipo de função (por exemplo, a fórmula de autoatenção mostrada acima, no entanto, essa função pode ser quase qualquer coisa) e, em seguida, a produzem. Os blocos geralmente são pilhas de camadas umas sobre as outras, fazendo algo semelhante a uma única camada, mas várias vezes.






Vamos começar examinando a Figura 1 do ViT Paper.

As principais coisas às quais prestaremos atenção são:

Camadas - recebe uma entrada, executa uma operação ou função na entrada, produz uma saída.
Blocos - uma coleção de camadas, que por sua vez também recebe uma entrada e produz uma saída.

![image](https://github.com/user-attachments/assets/4c96d4c4-9287-461f-8c66-933ef74a7464)











