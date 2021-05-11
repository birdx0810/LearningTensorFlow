# Deep Learning

## Introduction 

Deep Learning is a branch of machine learning that is based on artificial neural networks and gradient descent optimization to learn from data. The notion of machine learning could be traced back to Alan Turing's paper "Computing Machinery and Intelligence", stating that "...what we want is a machine that could learn from experience". 

> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. - Tom M. Mitchell

Methods and concepts are mostly similar to that of Data Mining and Pattern Recognition.

**Deep learning** could be categorized in to two objectives, _regression_, i.e. predicting values; and _classification_, predicting labels. 

**Learning** could be categorized into supervised, i.e. the labels are provided; unsupervised, i.e. labels are not provided; and reinforcement learning, i.e. a numerical score is provided to the model as a guidance. 

**Models** could be divided into generative, i.e. statistical model of the joint probability distribution on observed value $$x$$ and target variable $$y$$; and discriminative, i.e. statistical model of the conditional probability of target $$y$$ given an observation $$x$$.

## Loss Functions

### Mean Square Error

$$
\text{MSE}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n(y_i - \hat{y}_i)^2
$$

Where $$n$$ represents the number of features of the dataset. This loss function is highly associated with regression models, auto-encoders.

### Negative Log Likelihood

$$
\text{NLL}(p, y) = -\sum_{i=1}^n y_i\log(p_i)
$$

Here, $$n$$ represents the number of classes that are being predicted. This loss function is highly associated with classification tasks. For language models the classes are the words within the vocabulary; for images, it could be a particular label like "cat" or "dog".

### Contrastive Loss

The following loss is also known as Pairwise Ranking Loss.



$$
\text{PRL}(r_1, r_2) = \begin{cases}
d(r_1, r_2), \text{} \quad \quad \quad \quad \quad \quad \quad \text{if positive pair;}\\
\max \big(0, m - d(r_1, r_2) \big), \quad \text{ otherwise.}
\end{cases}
$$

Where $$d$$ is the distance function and $$m$$ is a margin value for negative pairs. There is a variant known as Triplets Ranking Loss is as the following.

$$
\text{TRL}(r, r_p, r_n) = \max\big( 0, m + d(r, r_p) - d(r, r_n) \big)
$$



