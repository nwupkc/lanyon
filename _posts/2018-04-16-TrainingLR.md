---
layout: post
title: Training Linear Regression
---


In this blog post, we will look at the Linear Regression model, one of the fundamental models. We will discuss two different ways to train it:

- Using a Normal equation that directly computes the model parameters that best fit the model to the training set by finding the model parameters that minimize the cost function.
- Using an iterative optimization approach called Gradient Descent that gradually tweaks the model parameters to minimize the cost function, converging to the same set of parameters as the Normal equation.

A linear model makes a prediction by simply computing a weighted sum of the input variables, plus a constant called the bias term or the intercept term, as shown below.

$$\hat y = \beta_0 +\beta_1 x_1 + \beta_2 x_2 + ⋯ + \beta_p x_p$$

- ŷ is the predicted value
- p is the number of variables
- $$x_i$$ is the $$i^{th}$$ feature value
- $$\beta_i$$ is the $$i^{th}$$ model parameter

This can be written much more concisely using a vectorized form, as shown below:

$$\hat y = \pmb\beta^T \cdot \pmb x = \
\begin{bmatrix} \beta_0 & \beta_1 & \cdots & \beta_p \end{bmatrix} \cdot \begin{bmatrix} 1 \\ x_1 \\ \vdots \\ x_p \end{bmatrix}$$

- $$\pmb\beta^T$$ is the transpose of $$\pmb\beta$$ which is a column vector of model parameters.
- $$\pmb x$$ is a column vector, where $$x_0$$ equals to 1.

## Normal Equation

In order to train a Linear Regression model, we need to find the value of $$\beta s$$ that minimizes the cost function, Mean-Squared-Error (MSE). MSE cost function for a Linear Regression model looks like this:

$$MSE = \frac{1}{n} \sum\limits_{i=1}^n (\hat y_i - y_i)^2 = \frac{1}{n} \sum\limits_{i=1}^n (\pmb\beta^T \cdot \pmb x - y_i)^2$$

We will define the "design matrix" $$\pmb X$$ as a matrix of n rows, in which each row is the $$i^{th}$$ sample (the vector $$\pmb{x_i}$$).

$$\pmb X = \begin{bmatrix} \pmb{x_{1}} \\ \pmb{x_{2}} \\
\vdots \\ \pmb{x_{n}} \end{bmatrix} \
= \begin{bmatrix} x_{10} & x_{11} & \cdots & x_{1p} \\
x_{20} & x_{21} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n0} & x_{n1} & \cdots & x_{np} \\
\end{bmatrix}$$

With this, we can rewrite the cost function as following, replacing the explicit sum by matrix multiplication:

$$MSE = \frac{1}{n} (\pmb X \pmb\beta - \pmb y)^T (\pmb X \pmb\beta - \pmb y)$$

where

$$\pmb\beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\
\vdots \\ \beta_p \end{bmatrix},
\pmb y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

Simplify:

$$MSE = \frac{1}{n} ((\pmb X \pmb\beta)^T - \pmb y^T) (\pmb X \pmb\beta - \pmb y)$$

$$= \frac{1}{n} (\pmb X \pmb\beta)^T \pmb X \pmb\beta \
- (\pmb X \pmb\beta)^T \pmb y \
- \pmb y^T (\pmb X \pmb\beta) \
+ \pmb y^T \pmb y)$$

Notice that $$\pmb{X\beta}$$ is a n x 1 vector since $$\pmb{X}$$ is n x p+1 matrix and $$\pmb\beta$$ is p+1 x 1 vector and y is also a n x 1 vector. Therefore, the order does not matter, and thus we can further simplify:

$$MSE = \frac{1}{n} \pmb\beta^T \pmb X^T  \pmb X \pmb\beta \
- 2(\pmb X \pmb\beta)^T \pmb y \
+ \pmb y^T \pmb y)$$

Take first order condition w.r.t. $$\pmb\beta$$ and set it equal to zero:

$$\frac{\partial MSE}{\partial \pmb \beta} = 2\pmb X^T \pmb X \pmb\beta - 2 \pmb X^T \pmb y = 0$$

$$\pmb X^T \pmb X \pmb\beta = \pmb X^T \pmb y$$

$$\pmb\beta = (\pmb X^T \cdot \pmb X)^{-1} \cdot \pmb X^T \cdot \pmb y$$

This is called a closed-form solution or the Normal Equation.

### Computational Complexity

It takes:

- $$O(p^2n)$$ to multiply $$\pmb{X^T}$$ by $$\pmb{X}$$
- $$O(pn)$$ to multiply $$\pmb{X^T}$$ by $$\pmb{y}$$
- $$O(p^3)$$ to invert $$\pmb{X^TX}$$

Asymptotically, $$O(p^2n)$$ dominates $$O(pn)$$ so we can forget the $$O(pn)$$ part. We will assume that n > p since otherwise the matrix $$\pmb{X^TX}$$ would be singular or degenerate (and hence non-invertible), which means that $$O(p^2n)$$ asymptotically dominates $$O(p^3)$$.

Therefore the total time complexity is $$O(p^2n)$$.

Disregard for now that n > p. The Normal Equation computes the inverse of $$\pmb{X^TX}$$ which has computational complexity is $$O(p^3)$$. This means that if you double the number of variables, you are multiplying the computation time by 8 times. Therefore, the Normal Equation gets very slow when the number of variables is large.

On the other hand, this equation is linear with regards to the number of observations in the training set (i.e., $$O(n)$$), so it handles large training sets efficiently, provided they can fit in memory.

Also, once you have trained your Linear Regression model, predictions are very fast: the computational complexity is linear with regards to both the number of observations you want to make predictions on and the number of variables since it takes $$O(pn)$$ to multiply $$\pmb{\beta}$$ by $$\pmb{X}$$. Making predictions on twice as many observations or twice as many variables will take twice as much time.

## Gradient Descent

Now we will consider a different way to train a Linear Regression model which is better suited for cases where there are a large number of variables, or too many training observations to fit in memory.

_Gradient Descent_ is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.

__What is Gradient__ the gradient is a multi-variable generalization of the derivative. While a derivative can be defined on functions of a single variable, for functions of several variables, the gradient takes its place. The gradient is a vector-valued function, as opposed to a derivative, which is scalar-valued.

Like the derivative, the gradient represents the slope of the tangent of the graph of the function. More precisely, the gradient points in the direction of the greatest rate of increase of the function, and its magnitude is the slope of the graph in that direction. The components of the gradient in coordinates are the coefficients of the variables in the equation of the tangent space to the graph.

Gradient Descent measures the local gradient of the error function with regards to the parameter vector $$\beta s$$, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum.

Concretely, you start by filling $$\beta$$ with random values (this is called random initialization), and then you improve it gradually, taking one baby step at a time, each step attempting to decrease the cost function (e.g., the MSE), until the algorithm converges to a minimum.

An important parameter in Gradient Descent is the size of the steps, determined by the learning rate hyper-parameter. If the learning rate is too small, then the algorithm will have to go through many iterations to converge, which will take a long time.

On the other hand, if the learning rate is too high, you might jump across the valley and end up on the other side, possibly even higher up than you were before. This might make the algorithm diverge, with larger and larger values, failing to find an optimal solution.

Convexity means that if you pick any two points on the curve, the line segment joining them never crosses the curve. This implies that there are no local minima, just one global minimum. It is also a continuous function with a slope that never changes abruptly. These two facts have a great consequence: Gradient Descent is guaranteed to approach arbitrarily close the global minimum (if you wait long enough and if the learning rate is not too high). Fortunately, the MSE cost function for a Linear Regression model happens to be a convex function.

But not all cost functions are nice convex functions. There may be holes, ridges, plateaus, and all sorts of irregular terrains, making convergence to the minimum very difficult. If the random initialization starts the algorithm on some places, then it will converge to a local minimum rather than the global minimum. If it starts on the plateau, it will take a very long time to cross and if you stop prematurely, you will never reach the global minimum.

When using Gradient Descent, you should ensure that all features have a similar scale or else it will take much longer to converge.

Training a model means searching for a combination of model parameters that minimizes a cost function. It is a search in the model’s parameter space: the more parameters a model has, the more dimensions this space has, and the harder the search is: searching for a needle in a 300-dimensional haystack is much trickier than in three dimensions.
