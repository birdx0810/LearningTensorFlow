# Image Recognition

## ImageNet \(ILSVRC\)

## Shallow Networks

### LeNet \([Lecun et al., 1998](https://ieeexplore.ieee.org/document/726791)\)

![](https://i.imgur.com/HGR9btg.png)

_**Layer Architecture**_ 

1. Conv1 + MeanPool1
2. Conv2 + MeanPool2
3. Flatten + FCL1 
4. FCL2 
5. FCL3

### AlexNet \([Krizhevsky et al., 2012](http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)\)

![](https://i.imgur.com/ILtk3Q8.png)

_**Layer Architecture**_ 

1. Conv1 + MaxPool
2. Conv2 + MaxPool
3. Conv3
4. Conv4
5. Conv5 + MaxPool
6. Flatten + FCL1
7. FCL2
8. FCL3

### ZFNet \([Zeiler and Fergus, 2013](https://arxiv.org/abs/1311.2901)\)

Modified AlexNet using deconvolution

![](https://i.imgur.com/g2id0sH.png)

## Deeper Networks



#### GoogLeNet \(2014\)

> Christian Szegedy
>
> **Inception\(v1\): Going Deeper with Convolutions**
>
> ILSVRC 2014/IEEE 2015 Paper: [Link](https://research.google/pubs/pub43022/)

![](https://i.imgur.com/LukLpPO.png)

Create filters with multiple sizes on the same level, making models "wider" instead of "deeper".

**Inception\(v2/v3\)**

> Paper: [Link](https://research.google/pubs/pub44903/)

**Inception\(v4\) & Inception-ResNet**

> Paper: [Link](https://research.google/pubs/pub45169/)

#### VGG16/19 \(2014\)

> Paper: [Link](https://arxiv.org/abs/1409.1556)

![](https://i.imgur.com/bn5Wbuo.png)

#### ResNet/ResNeXt \(2015\)

> He Kaiming Papers:
>
> * [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
> * [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
> * [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

As models get deeper, accuracy gets saturated, converges, and degrades rapidly due to overfitting.

#### DenseNet: Densely Connected Convolutional Networks \(2016\)

> Paper: [Link](https://arxiv.org/abs/1608.06993)

#### Xception: Deep Learning with Depthwise Separable Convolutions \(2017\)

> IEEE 2017 Paper: [Link](https://ieeexplore.ieee.org/document/8099678)

#### DarkNet \(2018\)

> See [YOLO](https://hackmd.io/T8wQunudSEG4VJAILspdlg?both#YOLO-You-Only-Look-Once)

