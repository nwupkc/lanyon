---
layout: post
title: Classification Challenges
---

**Topic: Classification**

**Settings: Where applicable, use `test_size=.30, random_state=4444`. This will permit comparison of results across users.

**Data:**

Challenges 1-10: congressional votes [Congressional Voting Records Dataset](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)

Challenge 11: movie data

Challenge 12: breast cancer surgery [Haberman Survival Dataset](https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival)

**Data – Congressional Votes**

Download the congressional votes data from here: [Congressional Voting Records Dataset](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)

These are votes of U.S. House of Representatives Congressmen on 16 key issues in 1984.

Read the description of the fields and download the data: house-votes-84.data

We will try to see if we can predict the house members' party based on their votes.

We will also use some of the general machine learning tools we learned (a bit more efficiently this time).

## Challenge 1

Load the data into a pandas dataframe. Replace 'y's with 1s, 'n's with 0s.

Now, almost every representative has a ?. This represents the absence of a vote (they were absent or some other similar reason). If we dropped all the rows that had a ?, we would throw out most of our data. Instead, we will replace ? with the best guess in the Bayesian sense: in the absence of any other information, we will say that the probability of the representative saying YES is the ratio of others that said YES over the whole votes.

So, convert each ? to this probability (when yes=1 and no=0, this is the mean of the column)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
df = pd.read_csv('/home/pk/data/house-votes-84.data')
```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>republican</th>
      <th>n</th>
      <th>y</th>
      <th>n.1</th>
      <th>y.1</th>
      <th>y.2</th>
      <th>y.3</th>
      <th>n.2</th>
      <th>n.3</th>
      <th>n.4</th>
      <th>y.4</th>
      <th>?</th>
      <th>y.5</th>
      <th>y.6</th>
      <th>y.7</th>
      <th>n.5</th>
      <th>y.8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>republican</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>democrat</td>
      <td>?</td>
      <td>y</td>
      <td>y</td>
      <td>?</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>democrat</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>?</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>democrat</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>?</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>democrat</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>republican</th>
      <th>n</th>
      <th>y</th>
      <th>n.1</th>
      <th>y.1</th>
      <th>y.2</th>
      <th>y.3</th>
      <th>n.2</th>
      <th>n.3</th>
      <th>n.4</th>
      <th>y.4</th>
      <th>?</th>
      <th>y.5</th>
      <th>y.6</th>
      <th>y.7</th>
      <th>n.5</th>
      <th>y.8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>429</th>
      <td>republican</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>430</th>
      <td>democrat</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>431</th>
      <td>republican</td>
      <td>n</td>
      <td>?</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>432</th>
      <td>republican</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>433</th>
      <td>republican</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>?</td>
      <td>n</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.replace('y', 1, inplace=True)
df.replace('n', 0, inplace=True)
```


```python
df.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>republican</th>
      <th>n</th>
      <th>y</th>
      <th>n.1</th>
      <th>y.1</th>
      <th>y.2</th>
      <th>y.3</th>
      <th>n.2</th>
      <th>n.3</th>
      <th>n.4</th>
      <th>y.4</th>
      <th>?</th>
      <th>y.5</th>
      <th>y.6</th>
      <th>y.7</th>
      <th>n.5</th>
      <th>y.8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>429</th>
      <td>republican</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>430</th>
      <td>democrat</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>431</th>
      <td>republican</td>
      <td>0</td>
      <td>?</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>432</th>
      <td>republican</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>433</th>
      <td>republican</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>?</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.replace('?', np.nan, inplace=True)
```


```python
df.mean()
```




    n      0.443128
    y      0.502591
    n.1    0.598109
    y.1    0.416076
    y.2    0.503580
    y.3    0.640662
    n.2    0.569048
    n.3    0.577566
    n.4    0.502427
    y.4    0.503513
    ?      0.362319
    y.5    0.421836
    y.6    0.508557
    y.7    0.592326
    n.5    0.428571
    y.8    0.812121
    dtype: float64




```python
df.fillna(df.mean(), inplace=True).head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>republican</th>
      <th>n</th>
      <th>y</th>
      <th>n.1</th>
      <th>y.1</th>
      <th>y.2</th>
      <th>y.3</th>
      <th>n.2</th>
      <th>n.3</th>
      <th>n.4</th>
      <th>y.4</th>
      <th>?</th>
      <th>y.5</th>
      <th>y.6</th>
      <th>y.7</th>
      <th>n.5</th>
      <th>y.8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>republican</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.812121</td>
    </tr>
    <tr>
      <th>1</th>
      <td>democrat</td>
      <td>0.443128</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.416076</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>democrat</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.50358</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>democrat</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.421836</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>democrat</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Challenge 2

Split the data into a test and training set. Use this function:


```python
from sklearn.model_selection import train_test_split
```


```python
y = df['republican']
X = df.drop('republican', axis=1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)
```

## Challenge 3

Using scikit.learn's KNN algorithm, train a model that predicts the party (republican/democrat):


```python
from sklearn.neighbors import KNeighborsClassifier
```

Try it with a lot of different k values (number of neighbors), from 1 to 20, and on the test set calculate the accuracy (number of correct predictions / number of all predictions) for each k

You can use this to calculate accuracy:


```python
from sklearn.metrics import accuracy_score
```

Which k value gives the highest accuracy?


```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.923664122137



```python
k_range = list(range(1, 21))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    k_scores.append((accuracy_score(y_test, y_pred)))
pd.DataFrame(k_scores, index=k_range, columns = ['accuracy'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.908397</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.908397</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.908397</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.916031</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.923664</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.908397</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.916031</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.916031</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.923664</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.908397</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.923664</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.931298</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.931298</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.916031</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.916031</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.916031</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.916031</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.908397</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.916031</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.900763</td>
    </tr>
  </tbody>
</table>
</div>



## Challenge 4

Make a similar model but with `LogisticRegression` instead, calculate test accuracy.


```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5, random_state=4444)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
l_score = accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
```

    0.954198473282


## Challenge 5

Make a bar graph of democrats and republicans. How many of each are there?

Make a very simple predictor that predicts 'democrat' for every incoming example.

Just make a function that takes in an X --an array or matrix with input examples--, and returns an array of the same length as X, where each value is 'democrat'. For example, if X is three rows, your function should return ['democrat','democrat','democrat']. Make a y_predicted vector using this and measure its accuracy.

Do the same with predicting 'republican' all the time and measure its accuracy.


```python
plt.style.use('seaborn-poster')
ax = y.value_counts().plot(kind='bar', color=('b', 'r'), rot='horizontal')
```


![png](/images/Classification_files/Classification_27_0.png)



```python
y.value_counts()
```




    democrat      267
    republican    167
    Name: republican, dtype: int64




```python
def democrat(X_test):
    return np.repeat('democrat', len(X_test))

y_pred = democrat(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.603053435115



```python
def republican(X_test):
    return np.repeat('republican', len(X_test))

y_pred = republican(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.396946564885


## Challenge 6

Plot the accuracies as a function of k. Since k only matters for KNN, your logistic regression accuracy, 'democrat' predictor accuracy and 'republican' predictor accuracy will stay the same over all k, so each of these three will be a horizontal line. But the KNN accuracy will change with k.


```python
# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.axhline(y=l_score, color='r', linestyle='-')
plt.title('Accuracies as a function of K')
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy')
plt.legend(['kNN', 'logreg'], loc=7);
```


![png](/images/Classification_files/Classification_32_0.png)


## Challenge 7

Plot a learning curve for the logistic regression model. But instead of going through the painstaking steps of doing it yourself, use this function:


```python
from sklearn.model_selection import learning_curve
```

This will give you the m, training and testing accuracy scores. All you need to do is plot them. You don't even need to give it separate training/test sets. It will do crossvalidation all by itself. Easy, isn't it? : ) Remember, since it does cross-validation, it doesn't have a single training or test accuracy per m value. Instead, it has one for each fold (separate partition) of the cross validation. A good idea is to take the mean of these scores from different folds. This gives you a meaningful single number per m. What I mean is that doing something like:

`train_cv_scr = np.mean(train_err, axis=1)
test_cv_scr = np.mean(ts_err, axis=1)`

Before plotting `m` vs `train_cv_scr` and `m` vs `test_cv_scr`, where `train_scores` and `test_scores` are the vectors returned by the learning curve function. The `np.mean(...., axis=1)` means take the mean along axis 1 (axis 1 is the columns axis-- for each row, you have a bunch of columns, each corresponding to a cross validation fold, you are averaging these columns for each row).

Draw the learning curve for KNN with the best k value as well.


```python
m_range = list(range(2, 21))
train_cv_scr = []
test_cv_scr = []
for m in m_range:
    train_sizes, train_scores, valid_scores = learning_curve(LogisticRegression(C=1e5, random_state=4444), X, y, train_sizes=[0.70], cv=m)
    train_cv_scr.append(np.mean(train_scores, axis=1))
    test_cv_scr.append(np.mean(valid_scores, axis=1))

plt.plot(m_range, train_cv_scr)
plt.plot(m_range, test_cv_scr, c='r')
plt.title('K vs CV Scores')
plt.xlabel('Value of K for K-Fold CV')
plt.ylabel('Accuracy Score')
plt.legend(['Train', 'Test'], loc=4);     
```


![png](/images/Classification_files/Classification_36_0.png)



```python
# Draw the learning curve for KNN with the best k value as well.
m_range = list(range(2, 21))
train_cv_scr = []
test_cv_scr = []
for m in m_range:
    train_sizes, train_scores, valid_scores = learning_curve(KNeighborsClassifier(n_neighbors=12), X, y, train_sizes=[0.70], cv=m)
    train_cv_scr.append(np.mean(train_scores, axis=1))
    test_cv_scr.append(np.mean(valid_scores, axis=1))

plt.plot(m_range, train_cv_scr)
plt.plot(m_range, test_cv_scr, c='r')
plt.title('K vs CV Scores')
plt.xlabel('Value of K for K-Fold CV')
plt.ylabel('Accuracy Score')
plt.legend(['Train', 'Test'], loc=4);
```


![png](/images/Classification_files/Classification_37_0.png)


## Challenge 8

This is a preview of many other classification algorithms that we will go over. Scikit.learn has the same interface for all of these, so you can use them exactly the same way as you did LogisticRegression and KNeighborsClassifier. Use each of these to classify your data and print the test accuracy of each:

Gaussian Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
```

SVM (Support Vector Machine) Classifier


```python
from sklearn.svm import SVC
```

Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
```

Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.908396946565



```python
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.954198473282



```python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.923664122137



```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.931297709924


## Challenge 9

There is actually a way to do cross validation quickly to get your accuracy results for an algorithm, without separating training and test yourself:


```python
from sklearn.model_selection import cross_val_score
```

Just like the `learning_curve` function, this takes a classifier object, `X` and `Y`. Returns accuracy (or whatever score you prefer by using the scoring keyword argument). Of course, it will return a score for each cross validation fold, so to get the generalized accuracy, you need to take the mean of what it returns.

Use this function to calculate the cross validation score of each of the classifiers you tried before.


```python
knn = KNeighborsClassifier(n_neighbors=5)
log = LogisticRegression(C=1e5, random_state=4444)
gnb = GaussianNB()
svc = SVC()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()

models = [knn, log, gnb, svc, tree, forest]
cv_scores = []
for m in models:
    cv_scores.append(np.mean(cross_val_score(m, X, y)))
pd.DataFrame(cv_scores, index=['knn', 'log', 'gnb', 'svc', 'tree', 'forest'], columns = ['cv_score'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cv_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>knn</th>
      <td>0.933190</td>
    </tr>
    <tr>
      <th>log</th>
      <td>0.960776</td>
    </tr>
    <tr>
      <th>gnb</th>
      <td>0.930843</td>
    </tr>
    <tr>
      <th>svc</th>
      <td>0.953879</td>
    </tr>
    <tr>
      <th>tree</th>
      <td>0.956178</td>
    </tr>
    <tr>
      <th>forest</th>
      <td>0.970019</td>
    </tr>
  </tbody>
</table>
</div>



## Challenge 10

Instead of 'democrat' or 'republican', can you predict the vote of a representative based on their other votes?

Reload the data from scratch. Convert y-->1, n-->0.

Choose one vote. Build a classifier (logistic regression or KNN), that uses the other votes (do not use the party as a feature) to predict if the vote will be 1 or 0.

Convert each ? to the mode of the column (if a senator has not voted, make their vote 1 if most others voted 1, make it 0 if most others voted 0).

Calculate the cross validation accuracy of your classifier for predicting how each representative will vote on the issue.


```python
df = pd.read_csv('/home/pk/data/house-votes-84.data')
df.replace('y', 1, inplace=True)
df.replace('n', 0, inplace=True)
```


```python
df.columns
```




    Index(['republican', 'n', 'y', 'n.1', 'y.1', 'y.2', 'y.3', 'n.2', 'n.3', 'n.4',
           'y.4', '?', 'y.5', 'y.6', 'y.7', 'n.5', 'y.8'],
          dtype='object')




```python
df.replace('?', np.nan, inplace=True)
df = df.fillna(df.mode().iloc[0])
```


```python
y = df['y.8']
X = df.drop(['republican', 'y.8'], axis=1)
```


```python
log = LogisticRegression(C=1e5, random_state=4444)
print(np.mean(cross_val_score(log, X, y)))
```

    0.834115581098


## Challenge 11

Back to movie data! Choose one categoric feature to predict. I chose MPAA Rating, but genre, month, etc. are all decent choices. If you don't have any non-numeric features, you can make two bins out of a numeric one (like "Runtime>100 mins" and "Runtime<=100 mins")

Make a bar graph of how many of each movie there is in the data. For example, with Ratings, show how many G, PG, PG-13, R movies there are, etc. (basically a histogram of your labels).

Predict your outcome variable (labels) using KNN and logistic regression. Calculate their accuracies.

Make a baseline stupid predictor that always predicts the label that is present the most in the data. Calculate its accuracy on a test set.

How much better do KNN and logistic regression do versus the baseline?

What are the coefficients of logistic regression? Which features affect the outcome how?


```python
movies = pd.read_csv('/home/pk/data/2013_movies.csv')
```


```python
movies.Rating.value_counts().plot(kind='bar', rot=0);
```


![png](/images/Classification_files/Classification_62_0.png)



```python
# for sklearn classifiers to work
movies = movies.dropna()
movies['ReleaseMonth'] = pd.to_datetime(movies['ReleaseDate']).dt.month
movies['ReleaseDay'] = pd.to_datetime(movies['ReleaseDate']).dt.day
movies = pd.concat([movies, pd.get_dummies(movies['Director'])], axis=1)
```


```python
y = movies['Rating']
X = movies.drop(['Rating', 'ReleaseDate', 'Director', 'Title'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)
```


```python
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.555555555556



```python
logreg = LogisticRegression(C=1e5, random_state=4444)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.444444444444



```python
y_pred = np.repeat('PG-13', len(X_test))
print(accuracy_score(y_test, y_pred))
```

    0.407407407407


## Challenge 12

Now you are a classification master. The representative votes dataset only had 0s and 1s. Let's just swiftly tackle the breast cancer surgery data.

Get it from here: [Haberman Survival Dataset](https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival)

- What is the average and standard deviation of the age of all of the patients?
- What is the average and standard deviation of the age of those patients that survived 5 or more years after surgery?
- What is the average and standard deviation of the age of those patients who survived fewer than 5 years after surgery?
- Plot a histogram of the ages side by side with a histogram of the number of axillary nodes.
- What is the earliest year of surgery in this dataset?
- What is the most recent year of surgery?
- Use logistic regression to predict survival after 5 years. How well does your model do?
- What are the coefficients of logistic regression? Which features affect the outcome how?
- Draw the learning curve for logistic regression in this case.


```python
haberman = pd.read_csv('/home/pk/data/haberman.data', header=None, names=['age', 'year', 'nodes', 'survival'])
```


```python
haberman.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>year</th>
      <th>nodes</th>
      <th>survival</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>306.000000</td>
      <td>306.000000</td>
      <td>306.000000</td>
      <td>306.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52.457516</td>
      <td>62.852941</td>
      <td>4.026144</td>
      <td>1.264706</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.803452</td>
      <td>3.249405</td>
      <td>7.189654</td>
      <td>0.441899</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>58.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>44.000000</td>
      <td>60.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>52.000000</td>
      <td>63.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.750000</td>
      <td>65.750000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>83.000000</td>
      <td>69.000000</td>
      <td>52.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
haberman[haberman.survival == 1].describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>year</th>
      <th>nodes</th>
      <th>survival</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>225.000000</td>
      <td>225.000000</td>
      <td>225.000000</td>
      <td>225.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52.017778</td>
      <td>62.862222</td>
      <td>2.791111</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.012154</td>
      <td>3.222915</td>
      <td>5.870318</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>58.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>43.000000</td>
      <td>60.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>52.000000</td>
      <td>63.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.000000</td>
      <td>69.000000</td>
      <td>46.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
haberman[haberman.survival == 2].describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>year</th>
      <th>nodes</th>
      <th>survival</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.679012</td>
      <td>62.827160</td>
      <td>7.456790</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.167137</td>
      <td>3.342118</td>
      <td>9.185654</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>34.000000</td>
      <td>58.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>46.000000</td>
      <td>59.000000</td>
      <td>1.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>53.000000</td>
      <td>63.000000</td>
      <td>4.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.000000</td>
      <td>65.000000</td>
      <td>11.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>83.000000</td>
      <td>69.000000</td>
      <td>52.000000</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
haberman.hist(column=['age','nodes']);
```


![png](/images/Classification_files/Classification_73_0.png)



```python
haberman.year.min() + 1900
```




    1958




```python
haberman.year.max() + 1900
```




    1969




```python
y = haberman['survival']
X = haberman.drop('survival', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)
```


```python
logreg = LogisticRegression(C=1e5, random_state=4444)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.836956521739



```python
logreg.coef_
```




    array([[ 0.01747104,  0.00634131,  0.0719581 ]])




```python
m_range = list(range(2, 21))
train_cv_scr = []
test_cv_scr = []
for m in m_range:
    train_sizes, train_scores, valid_scores = learning_curve(LogisticRegression(C=1e5, random_state=4444), X, y, train_sizes=[0.70], cv=m)
    train_cv_scr.append(np.mean(train_scores, axis=1))
    test_cv_scr.append(np.mean(valid_scores, axis=1))

plt.plot(m_range, train_cv_scr)
plt.plot(m_range, test_cv_scr, c='r')
plt.title('K vs CV Scores')
plt.xlabel('Value of K for K-Fold CV')
plt.ylabel('Accuracy Score')
plt.legend(['Train', 'Test'], loc=4);
```


![png](/images/Classification_files/Classification_79_0.png)
