---
layout: post
title: Linear Regression and Learning Curve Challenges
---

### Challenge 1
Generate (fake) data that is linearly related to $$log(x)$$.

You are making this model up. It is of the form $$\beta_{0} + \beta_{1}log(x) + \epsilon$$. (You are making up the parameters.)

Simulate some data from this model.

Then fit two models to it:

* quadratic (second degree polynomial)
* logarithmic ($$log(x)$$)

(The second one should fit really well, since it has the same form as the underlying model!)


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import patsy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```


```python
np.random.seed(11)
def f(x):
    return np.log(x) + 5
X = np.random.uniform(0, 1, size = 100)[:, np.newaxis]  # force matrix
y = f(X) + np.random.normal(scale = 0.3, size = 100)[:, np.newaxis]
```


```python
degree = 2
est = make_pipeline(PolynomialFeatures(degree), LinearRegression(fit_intercept = False))
est.fit(X, y)
est.score(X, y)
```




    0.8749158343991223




```python
log_X = np.log(X)
lr = LinearRegression()
lr.fit(log_X, y)
lr.score(log_X, y)
```




    0.9342794210170648



### Challenge 2
Generate (fake) data from a model of the form $$\beta_{0} + \beta_{1}x + \beta_{2} x^2 + \epsilon$$. (You are making up the parameters.)

Split the data into a training and test set.

Fit a model to your training set. Calculate mean squared error on your training set. Then calculate it on your test set.

(You could use `sklearn.metrics.mean_squared_error`.)


```python
np.random.seed(11)
def f(x):
    return x ** 2 + 3 * x + 5
X = np.random.uniform(0, 1, size = 100)[:, np.newaxis]  # force matrix
y = f(X) + np.random.normal(scale = 0.3, size = 100)[:, np.newaxis]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
degree = 2
est = make_pipeline(PolynomialFeatures(degree), LinearRegression(fit_intercept = False))
est.fit(X_train, y_train)
train_error = mean_squared_error(y_train, est.predict(X_train))
test_error = mean_squared_error(y_test, est.predict(X_test))
train_error
```




    0.06333832121141522




```python
test_error
```




    0.11073858315314192



### Challenge 3
For the data from two (above), try polynomial fits from 0th (just constant) to 7th order (highest term $$x^7$$). Over the x axis of model degree (8 points), plot:

* training error
* test error
* $$R^2$$
* AIC


```python
def plot_approximation(est, ax, label = None):
    """Plot the approximation of ``est`` on axis ``ax``. """
    ax.plot(x_plot, f(x_plot), label = 'ground truth', color = 'green')
    ax.scatter(X, y, s = 100)   # our data
    ax.plot(x_plot, est.predict(x_plot[:, np.newaxis]), color = 'red', label = label)
    ax.set_ylim((-2, 2))
    ax.set_xlim((0, 1))
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.legend(loc = 'upper right', frameon = True)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
train_error = np.empty(8)
test_error = np.empty(8)
r2 = np.empty(8)
aic = np.empty(8)
for degree in range(8):
    est = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    est.fit(X_train, y_train)
    train_error[degree] = mean_squared_error(y_train, est.predict(X_train))
    test_error[degree] = mean_squared_error(y_test, est.predict(X_test))
    r2[degree] = est.score(X_test, y_test)
    aic[degree] = sum((y_test - est.predict(X_test)) ** 2)


# Plot the training and test errors against degree
plt.figure(figsize = (8,6))
plt.plot(np.arange(8), train_error, color = 'green', label = 'train')
plt.plot(np.arange(8), test_error, color = 'red', label = 'test')
plt.plot(np.arange(8), r2, color = 'blue', label = 'r2')
plt.plot(np.arange(8), aic, color = 'black', label = 'aic')
plt.ylim((0.0, 5))
plt.xlabel('degree')
plt.legend(loc = 'upper left');
```


![png](/images/LearningCurve_files/LearningCurve_13_0.png)


### Challenge 4
For the data from two (above), fit a model to only the first 5 of your data points (m=5). Then to first 10 (m=10). Then to first 15 (m=15). In this manner, keep fitting until you fit your entire training set. For each step, calculate the training error and the test error. Plot both (in the same plot) over m. This is called a learning curve.


```python
train_error = np.empty(20)
test_error = np.empty(20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
for i in range(5, 100, 5):
    X_train2, y_train2 = X_train[:i], y_train[:i]
    est = make_pipeline(PolynomialFeatures(2), LinearRegression())
    est.fit(X_train2, y_train2)
    train_error[i//5] = mean_squared_error(y_train2, est.predict(X_train2))
    test_error[i//5] = mean_squared_error(y_test, est.predict(X_test))

# Plot the training and test errors against degree
plt.figure(figsize = (8,6))
plt.plot(np.arange(20), train_error, color = 'green', label = 'train')
plt.plot(np.arange(20), test_error, color = 'red', label = 'test')
plt.ylim((0.0, 0.3))
plt.ylabel('mean squared error')
plt.xlabel('training size')
plt.legend(loc = 'upper left');
```


![png](/images/LearningCurve_files/LearningCurve_15_0.png)
