# Tessarine-and-Quaternion-Valued-Convolutional-Neural-Networks
Basic foundations to emulate tessarine and quaternion valued CNNs on tensorflow.

Includes:
- Quaternion based weight initializer
- Hypercomplex Batch Normalization to 4 dimensional hypercomplex algebraic systems (e.g. tessarines and quaternions) layer
- Tessarine valued 2D convolution layer
- Tessarine valued residual unit 
- Quaternion valued 2D convolution layer
- Quaternion valued residual unit
- Real valued residual unit
- Example notebooks used for testing purposes

Used packages versions:
- tensorflow 2.5
- keras 2.4
- numpy 1.19
- scikit-learn 0.22

If using this code, please cite

@INPROCEEDINGS{senna2021tessarine,

AUTHOR = "Fernando Senna and Marcos Eduardo Valle",
    
TITLE = "Tessarine and Quaternion-Valued Deep Neural Networks for Image Classification",
    
BOOKTITLE = "Anais do XVIII Encontro Nacional de Inteligência Artificial e Computacional",

LOCATION = "Online Event",
    
DAYS = "29-3",
    
MONTH = "nov",
    
YEAR = "2021",

PAGES = "350--361",

PUBLISHER = "SBC",
 
ADDRESS = "Porto Alegre, RS, Brasil",

DOI = "10.5753/eniac.2021.18266",

URL = "https://sol.sbc.org.br/index.php/eniac/article/view/18266",
    
ABSTRACT = "Many image processing and analysis tasks are performed with deep neural networks. Although the vast majority of advances have been made with real numbers, recent works have shown that complex and hypercomplex-valued networks may achieve better results. In this paper, we address quaternion-valued and introduce tessarine-valued deep neural networks, including tessarine-valued 2D convolutions. We also address initialization schemes and hypercomplex batch normalization. Finally, a tessarine-valued ResNet model with hypercomplex batch normalization outperformed the corresponding real and quaternion-valued networks on the CIFAR dataset.",
    
KEYWORDS = "Artificial Neural Networks; Computational Intelligence; Computer Vision; Deep Learning"

}
