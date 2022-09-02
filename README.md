# sparse_CNNs
This project models Convolutional Neural Networks through sparse linear layers in order to explore the importance of the unique properties in convolutions.

We assess and compare the performance of four CNN LeNet (2 CNN layers, 3 FC layers) models when trained and tested on the Fashion-MNIST data set. 
One model is the 'GoldStandard', where the two CNN layers are implemented through PyTorch's built-in nn.Conv2D module. The other three models construct the CNN layers
by implementing the 'normal', 'shuffle', and 'scatter' schematics outlined in layers/models.py



