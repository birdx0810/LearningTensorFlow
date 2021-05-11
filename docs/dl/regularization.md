# Regularization

_**Definition:**_  
Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error. -- Ian Goodfellow \(2016\)

This happens due to the model architecture being too complex \(too many hyper-parameters\), and the model overfits to the training data. 

## Earlier Methods

* Early stopping: Stop training when validation performance gets worse.
* L1, L2 regularization: Where the lamda constraints the model parameters from overfitting
  * $$L_1\text{-norm}(w) = \text{Loss}(p,y) + \lambda\sum_{i=1}^n |w_i|$$
  * $$L_2\text{-norm}(w) = \text{Loss}(p,y) + \lambda\sum_{i=1}^n w^2_i$$
* Max-norm constraint: Where the model weights are restricted to $$||w||_2 < c$$.

## Dropout \([Srivastava et al., 2014](https://dl.acm.org/doi/abs/10.5555/2627435.2670313)\)

A probability $$p$$ of the model inputs are converted to $$0$$, thus ignoring the calculations for the dropped out neurons/weights. Variants include DropConnect \([Wan et al., 2013](http://proceedings.mlr.press/v28/wan13.html)\).

![](../.gitbook/assets/image%20%281%29.png)

## Batch Norm

Batch normalization scales each dimension of an input $$x$$ to a succeeding layer such that they have mean zero \($$\mu=0$$\) and unary variance \($$\sigma^2=1$$\), this is often used with CNNs.

![](../.gitbook/assets/image%20%282%29.png)

## Layer Norm

Layer normalization scales input $$x$$ along the feature dimension based on the other features, this is often used with RNNs.





