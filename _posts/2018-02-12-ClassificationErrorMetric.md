---
layout: post
title: Classification Error Metric Challenges
---

**Classification Error Metric Challenges**

**Settings: Where applicable, use `test_size=0.30, random_state=4444`. This will permit comparison of results across users.

These reference the Classification Challenges.

## Challenge 1
For the house representatives data set, calculate the accuracy, precision, recall and f1 scores of each classifier you built (on the test set).


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,precision_recall_fscore_support, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```


```python
df = pd.read_csv('/home/pk/data/house-votes-84.data')
df.replace('y', 1, inplace=True)
df.replace('n', 0, inplace=True)
df.replace('?', np.nan, inplace=True)
df.fillna(df.mean(), inplace=True).head()
df.replace('republican', 1, inplace=True)
df.replace('democrat', 0, inplace=True)
y = df['republican']
X = df.drop('republican', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)
```


```python
model1 = KNeighborsClassifier(n_neighbors=12)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
fpr1, tpr1, _ = roc_curve(y_test, y_pred)
```

    0.931297709924
    0.921568627451
    0.903846153846
    0.912621359223



```python
model2 = LogisticRegression(C=1e5, random_state=4444)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
fpr2, tpr2, _ = roc_curve(y_test, y_pred)
```

    0.954198473282
    0.96
    0.923076923077
    0.941176470588



```python
model3 = GaussianNB()
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
fpr3, tpr3, _ = roc_curve(y_test, y_pred)
```

    0.908396946565
    0.916666666667
    0.846153846154
    0.88



```python
model4 = SVC(probability=True)
model4.fit(X_train, y_train)
y_pred = model4.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
fpr4, tpr4, _ = roc_curve(y_test, y_pred)
```

    0.954198473282
    0.942307692308
    0.942307692308
    0.942307692308



```python
model5 = DecisionTreeClassifier()
model5.fit(X_train, y_train)
y_pred = model5.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
fpr5, tpr5, _ = roc_curve(y_test, y_pred)
```

    0.916030534351
    0.901960784314
    0.884615384615
    0.893203883495



```python
model6 = RandomForestClassifier()
model6.fit(X_train, y_train)
y_pred = model6.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
fpr6, tpr6, _ = roc_curve(y_test, y_pred)
```

    0.946564885496
    0.941176470588
    0.923076923077
    0.932038834951


## Challenge 2
For each, draw the ROC curve and calculate the AUC.


```python
import scikitplot as skplt

y_probas1 = model1.predict_proba(X_test)
y_probas2 = model2.predict_proba(X_test)
y_probas3 = model3.predict_proba(X_test)
y_probas4 = model4.predict_proba(X_test)
y_probas5 = model5.predict_proba(X_test)
y_probas6 = model6.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_probas1, figsize=(12, 8))
skplt.metrics.plot_roc_curve(y_test, y_probas2, figsize=(12, 8))
skplt.metrics.plot_roc_curve(y_test, y_probas3, figsize=(12, 8))
skplt.metrics.plot_roc_curve(y_test, y_probas4, figsize=(12, 8))
skplt.metrics.plot_roc_curve(y_test, y_probas5, figsize=(12, 8))
skplt.metrics.plot_roc_curve(y_test, y_probas6, figsize=(12, 8));
```


![png](/images/ClassificationErrorMetric_files/ClassificationErrorMetric_12_0.png)



![png](/images/ClassificationErrorMetric_files/ClassificationErrorMetric_12_1.png)



![png](/images/ClassificationErrorMetric_files/ClassificationErrorMetric_12_2.png)



![png](/images/ClassificationErrorMetric_files/ClassificationErrorMetric_12_3.png)



![png](/images/ClassificationErrorMetric_files/ClassificationErrorMetric_12_4.png)



![png](/images/ClassificationErrorMetric_files/ClassificationErrorMetric_12_5.png)


## Challenge 3
Calculate the same metrics you did in challenge 1, but this time in a cross validation scheme with the `cross_val_score` function (like in Challenge 9).


```python
knn = KNeighborsClassifier(n_neighbors=12)
log = LogisticRegression(C=1e5, random_state=4444)
gnb = GaussianNB()
svc = SVC()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()

models = [knn, log, gnb, svc, tree, forest]
scores = []
for m in models:
    scores.append((np.mean(cross_val_score(m, X, y)), np.mean(cross_val_score(m, X, y, scoring='precision')),
                   np.mean(cross_val_score(m, X, y, scoring='recall')), np.mean(cross_val_score(m, X, y, scoring='f1'))))
pd.DataFrame(scores, index=['knn', 'log', 'gnb', 'svc', 'tree', 'forest'], columns = ['accuracy', 'precision', 'recall', 'f1'])
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
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>knn</th>
      <td>0.926261</td>
      <td>0.881330</td>
      <td>0.934199</td>
      <td>0.906890</td>
    </tr>
    <tr>
      <th>log</th>
      <td>0.960776</td>
      <td>0.948990</td>
      <td>0.952165</td>
      <td>0.949724</td>
    </tr>
    <tr>
      <th>gnb</th>
      <td>0.930843</td>
      <td>0.907820</td>
      <td>0.916342</td>
      <td>0.910600</td>
    </tr>
    <tr>
      <th>svc</th>
      <td>0.953879</td>
      <td>0.921588</td>
      <td>0.964177</td>
      <td>0.941753</td>
    </tr>
    <tr>
      <th>tree</th>
      <td>0.946935</td>
      <td>0.937085</td>
      <td>0.952165</td>
      <td>0.949614</td>
    </tr>
    <tr>
      <th>forest</th>
      <td>0.963075</td>
      <td>0.942850</td>
      <td>0.976082</td>
      <td>0.942024</td>
    </tr>
  </tbody>
</table>
</div>



## Challenge 4
For your movie classifiers, calculate the precision and recall for each class.


```python
movies = pd.read_csv('/home/pk/data/2013_movies.csv')
movies = movies.dropna()
movies['ReleaseMonth'] = pd.to_datetime(movies['ReleaseDate']).dt.month
movies['ReleaseDay'] = pd.to_datetime(movies['ReleaseDate']).dt.day
movies = pd.concat([movies, pd.get_dummies(movies['Director'])], axis=1)
y = movies['Rating']
X = movies.drop(['Rating', 'ReleaseDate', 'Director', 'Title'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)
```


```python
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
pd.DataFrame([i for i in precision_recall_fscore_support(y_test, y_pred, labels=['PG-13', 'PG', 'R'])[:-1]],
             index=['precision', 'recall', 'f1'], columns=['PG-13', 'PG', 'R'])
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
      <th>PG-13</th>
      <th>PG</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.727273</td>
      <td>0.200000</td>
      <td>0.545455</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.592593</td>
      <td>0.285714</td>
      <td>0.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
log = LogisticRegression(C=1e5, random_state=4444)
log.fit(X_train, y_train)
y_pred = log.predict(X_test)
pd.DataFrame([i for i in precision_recall_fscore_support(y_test, y_pred, labels=['PG-13', 'PG', 'R'])[:-1]],
             index=['precision', 'recall', 'f1'], columns=['PG-13', 'PG', 'R'])
```

    /home/pk/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)





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
      <th>PG-13</th>
      <th>PG</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.409091</td>
      <td>0.0</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.818182</td>
      <td>0.0</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.545455</td>
      <td>0.0</td>
      <td>0.375000</td>
    </tr>
  </tbody>
</table>
</div>




```python
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
pd.DataFrame([i for i in precision_recall_fscore_support(y_test, y_pred, labels=['PG-13', 'PG', 'R'])[:-1]],
             index=['precision', 'recall', 'f1'], columns=['PG-13', 'PG', 'R'])
```

    /home/pk/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)





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
      <th>PG-13</th>
      <th>PG</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.363636</td>
      <td>0.0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.363636</td>
      <td>0.0</td>
      <td>0.727273</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.363636</td>
      <td>0.0</td>
      <td>0.592593</td>
    </tr>
  </tbody>
</table>
</div>




```python
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
pd.DataFrame([i for i in precision_recall_fscore_support(y_test, y_pred, labels=['PG-13', 'PG', 'R'])[:-1]],
             index=['precision', 'recall', 'f1'], columns=['PG-13', 'PG', 'R'])
```

    /home/pk/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)





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
      <th>PG-13</th>
      <th>PG</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.407407</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.578947</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
pd.DataFrame([i for i in precision_recall_fscore_support(y_test, y_pred, labels=['PG-13', 'PG', 'R'])[:-1]],
             index=['precision', 'recall', 'f1'], columns=['PG-13', 'PG', 'R'])
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
      <th>PG-13</th>
      <th>PG</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.363636</td>
      <td>0.0</td>
      <td>0.545455</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.347826</td>
      <td>0.0</td>
      <td>0.480000</td>
    </tr>
  </tbody>
</table>
</div>




```python
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
pd.DataFrame([i for i in precision_recall_fscore_support(y_test, y_pred, labels=['PG-13', 'PG', 'R'])[:-1]],
             index=['precision', 'recall', 'f1'], columns=['PG-13', 'PG', 'R'])
```

    /home/pk/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)





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
      <th>PG-13</th>
      <th>PG</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.375000</td>
      <td>0.0</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.545455</td>
      <td>0.0</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.444444</td>
      <td>0.0</td>
      <td>0.454545</td>
    </tr>
  </tbody>
</table>
</div>



## Challenge 5
Draw the ROC curve (and calculate AUC) for the logistic regression classifier from challenge 12.


```python
haberman = pd.read_csv('/home/pk/data/haberman.data', header=None, names=['age', 'year', 'nodes', 'survival'])
y = haberman['survival']
X = haberman.drop('survival', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)
logreg = LogisticRegression(C=1e5, random_state=4444)
logreg.fit(X_train, y_train)
y_probas = logreg.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_probas, figsize=(12, 8));
```


![png](/images/ClassificationErrorMetric_files/ClassificationErrorMetric_24_0.png)
