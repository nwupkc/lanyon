---
layout: post
title: Regularization
---


__Why Regularization:__ Regularizing (or shrinking) the coefficients estimates towards zero can significantly reduce their variance and thus fix overfitting. Overfitting occurs when there is low bias and high variance. Complex models can lead to overfitting the data, which means they follow the errors, or noise in data, too closely. This results in low training error but high testing error. Consequently, overfitting results in poor predictions on unseen data. What regularization does is that it awards goodness of fit while penalizing model complexity.

## Ridge Regression

Recall that the least squares fitting procedures estimates $$\beta$$s by minimizing:

$$ RSS = \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 $$

In contrast, the ridge regression coefficient estimates $$\hat\beta^R$$ are the values that minimize:

$$ \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda\sum_{j=1}^p \beta_j^2 = RSS + \lambda\sum_{j=1}^p \beta_j^2 $$

where $$\lambda\ge0$$ is a tuning parameter.

The difference, $$\lambda\sum_{j=1}^p \beta_j^2$$, is a shrinkage penalty (also called $$l_2$$ norm) which essentially penalizes having high $$\beta$$s. When $$\lambda=0$$ our ridge regression is equivalent to the least square regression. As $$\lambda$$ increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias. When $$\lambda\rightarrow\infty$$, the ridge regression coefficient estimates will approach zero. Notice that the shrinkage penalty is not applied to $$\beta_0$$. This is because we do not want to shrink the intercept, which is the mean output when $$x_{i1} = x_{i2} = \ldots = x_{ip} = 0$$.

## The Lasso

__Why Lasso__ One caveat of using the ridge regression is that it will use all p predictors when building model. The penalty $$\lambda\sum_{j=1}^p \beta_j^2$$ will shrink all of the coefficients towards zero, but it will not set any of them exactly to zero. The lasso regression overcomes this disadvantage. The lasso coefficients, $$\hat\beta^L$$, minimize the quantity:

$$ \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda\sum_{j=1}^p |\beta_j| = RSS + \lambda\sum_{j=1}^p |\beta_j| $$

The lasso uses an $$l_1$$ penalty, $$\lambda\sum_{j=1}^p |\beta_j|$$, while the ridge uses an $$l_2$$ penalty.
In the case of the lasso, the $$l_1$$ penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter $$\lambda$$ is sufficiently large. The lasso performs variable selection and thus yields sparse model meaning model involve only a subset of the variables. This gives us a huge advantage in model interpretation.

Again, when $$\lambda=0$$, the lasso simply gives the least squares fit, and when $$\lambda$$ becomes sufficiently large, the lasso gives the null model in which all coefficient estimates equal to zero.

## The Elastic Net

The elastic net is a regularized regression method that linearly combines the $$l_1$$ and $$l_2$$ penalties of the lasso and ridge methods. It uses both $$l_1$$ and $$l_2$$ penalties. The elastic net coefficients, $$\hat\beta^E$$ minimize the quantity:

$$ \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda(\alpha\sum_{j=1}^p |\beta_j| + (1-\alpha)\sum_{j=1}^p \beta_j^2) \\ = RSS + \lambda(\alpha\sum_{j=1}^p |\beta_j| + (1-\alpha)\sum_{j=1}^p \beta_j^2) $$

where $$\alpha\in[0,1]$$.
