---
layout: post
title: Linear Regression and Train/Test Split Challenges
---

### Challenge 1
Build a linear model that uses only a constant term (a column of ones) to predict a continuous outcome (like domestic total gross). How can you interpret the results of this model? What does it predict? Make a plot of predictions against actual outcome. Make a histogram of residuals. How are the residuals distributed?


```python
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import patsy
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```



```python
# load data
df = pd.read_csv("~/your_filepath/2013_movies.csv")
```


```python
# denominate Budget and Domestic Total Gross to in millions
df.Budget = df.Budget / 1000000
df.DomesticTotalGross = df.DomesticTotalGross / 1000000
df.head().round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Budget</th>
      <th>Domestic Total Gross</th>
      <th>Director</th>
      <th>Rating</th>
      <th>Run Time</th>
      <th>ReleaseDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Hunger Games: Catching Fire</td>
      <td>130.0</td>
      <td>424.7</td>
      <td>Francis Lawrence</td>
      <td>PG-13</td>
      <td>146</td>
      <td>2013-11-22 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iron Man 3</td>
      <td>200.0</td>
      <td>409.0</td>
      <td>Shane Black</td>
      <td>PG-13</td>
      <td>129</td>
      <td>2013-05-03 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>150.0</td>
      <td>400.7</td>
      <td>Chris BuckJennifer Lee</td>
      <td>PG</td>
      <td>108</td>
      <td>2013-11-22 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Despicable Me 2</td>
      <td>76.0</td>
      <td>368.1</td>
      <td>Pierre CoffinChris Renaud</td>
      <td>PG</td>
      <td>98</td>
      <td>2013-07-03 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Man of Steel</td>
      <td>225.0</td>
      <td>291.0</td>
      <td>Zack Snyder</td>
      <td>PG-13</td>
      <td>143</td>
      <td>2013-06-14 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create a column of ones
df["ones"] = 1
```


```python
df.head().round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Budget</th>
      <th>Domestic Total Gross</th>
      <th>Director</th>
      <th>Rating</th>
      <th>Run Time</th>
      <th>ReleaseDate</th>
      <th>ones</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Hunger Games: Catching Fire</td>
      <td>130.0</td>
      <td>424.7</td>
      <td>Francis Lawrence</td>
      <td>PG-13</td>
      <td>146</td>
      <td>2013-11-22 00:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iron Man 3</td>
      <td>200.0</td>
      <td>409.0</td>
      <td>Shane Black</td>
      <td>PG-13</td>
      <td>129</td>
      <td>2013-05-03 00:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>150.0</td>
      <td>400.7</td>
      <td>Chris BuckJennifer Lee</td>
      <td>PG</td>
      <td>108</td>
      <td>2013-11-22 00:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Despicable Me 2</td>
      <td>76.0</td>
      <td>368.1</td>
      <td>Pierre CoffinChris Renaud</td>
      <td>PG</td>
      <td>98</td>
      <td>2013-07-03 00:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Man of Steel</td>
      <td>225.0</td>
      <td>291.0</td>
      <td>Zack Snyder</td>
      <td>PG-13</td>
      <td>143</td>
      <td>2013-06-14 00:00:00</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y, X = patsy.dmatrices('DomesticTotalGross ~ ones', data = df, return_type = "dataframe")
model = sm.OLS(y, X)
fit = model.fit()
fit.summary()
```

    /Users/sungwankim/anaconda2/envs/python3/lib/python3.6/site-packages/statsmodels/regression/linear_model.py:1396: RuntimeWarning: divide by zero encountered in double_scalars
      return self.ess/self.df_model





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>DomesticTotalGross</td> <th>  R-squared:         </th> <td>  -0.000</td>
</tr>
<tr>
  <th>Model:</th>                    <td>OLS</td>        <th>  Adj. R-squared:    </th> <td>  -0.000</td>
</tr>
<tr>
  <th>Method:</th>              <td>Least Squares</td>   <th>  F-statistic:       </th> <td>    -inf</td>
</tr>
<tr>
  <th>Date:</th>              <td>Mon, 19 Feb 2018</td>  <th>  Prob (F-statistic):</th>  <td>   nan</td>
</tr>
<tr>
  <th>Time:</th>                  <td>14:37:44</td>      <th>  Log-Likelihood:    </th> <td> -588.44</td>
</tr>
<tr>
  <th>No. Observations:</th>       <td>   100</td>       <th>  AIC:               </th> <td>   1179.</td>
</tr>
<tr>
  <th>Df Residuals:</th>           <td>    99</td>       <th>  BIC:               </th> <td>   1181.</td>
</tr>
<tr>
  <th>Df Model:</th>               <td>     0</td>       <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>       <td>nonrobust</td>     <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   50.2984</td> <td>    4.370</td> <td>   11.510</td> <td> 0.000</td> <td>   41.628</td> <td>   58.969</td>
</tr>
<tr>
  <th>ones</th>      <td>   50.2984</td> <td>    4.370</td> <td>   11.510</td> <td> 0.000</td> <td>   41.628</td> <td>   58.969</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>51.114</td> <th>  Durbin-Watson:     </th> <td>   0.013</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 125.961</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.985</td> <th>  Prob(JB):          </th> <td>4.45e-28</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.804</td> <th>  Cond. No.          </th> <td>7.24e+16</td>
</tr>
</table>



The coefficient of the *ones* variable is the mean of the DomesticTotalGross.


```python
plt.style.use('seaborn-poster')
plt.plot(fit.predict(X), y, marker = '.', ls = 'None')
plt.title('Predictions Against Actual Outcome')
plt.xlabel('Prediction')
plt.ylabel('Actual');
```


![png](/images/TrainTestSplit_files/TrainTestSplit_9_0.png)



```python
fit.resid.hist(grid = False)
plt.title('histogram of residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency');
```


![png](/images/TrainTestSplit_files/TrainTestSplit_10_0.png)


Residuals are not normally distributed.

### Challenge 2
Repeat the process of challenge one, but also add one continuous (numeric) predictor variable. Also add plots of model prediction against your feature variable and residuals against feature variable. How can you interpret what's happening in the model?


```python
df.head().round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Budget</th>
      <th>Domestic Total Gross</th>
      <th>Director</th>
      <th>Rating</th>
      <th>Run Time</th>
      <th>ReleaseDate</th>
      <th>ones</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Hunger Games: Catching Fire</td>
      <td>130.0</td>
      <td>424.7</td>
      <td>Francis Lawrence</td>
      <td>PG-13</td>
      <td>146</td>
      <td>2013-11-22 00:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iron Man 3</td>
      <td>200.0</td>
      <td>409.0</td>
      <td>Shane Black</td>
      <td>PG-13</td>
      <td>129</td>
      <td>2013-05-03 00:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>150.0</td>
      <td>400.7</td>
      <td>Chris BuckJennifer Lee</td>
      <td>PG</td>
      <td>108</td>
      <td>2013-11-22 00:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Despicable Me 2</td>
      <td>76.0</td>
      <td>368.1</td>
      <td>Pierre CoffinChris Renaud</td>
      <td>PG</td>
      <td>98</td>
      <td>2013-07-03 00:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Man of Steel</td>
      <td>225.0</td>
      <td>291.0</td>
      <td>Zack Snyder</td>
      <td>PG-13</td>
      <td>143</td>
      <td>2013-06-14 00:00:00</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y, X = patsy.dmatrices('DomesticTotalGross ~ ones + Budget', data = df, return_type = "dataframe")
model = sm.OLS(y, X)
fit = model.fit()
fit.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>DomesticTotalGross</td> <th>  R-squared:         </th> <td>   0.286</td>
</tr>
<tr>
  <th>Model:</th>                    <td>OLS</td>        <th>  Adj. R-squared:    </th> <td>   0.278</td>
</tr>
<tr>
  <th>Method:</th>              <td>Least Squares</td>   <th>  F-statistic:       </th> <td>   34.82</td>
</tr>
<tr>
  <th>Date:</th>              <td>Mon, 19 Feb 2018</td>  <th>  Prob (F-statistic):</th> <td>6.80e-08</td>
</tr>
<tr>
  <th>Time:</th>                  <td>14:37:45</td>      <th>  Log-Likelihood:    </th> <td> -508.48</td>
</tr>
<tr>
  <th>No. Observations:</th>       <td>    89</td>       <th>  AIC:               </th> <td>   1021.</td>
</tr>
<tr>
  <th>Df Residuals:</th>           <td>    87</td>       <th>  BIC:               </th> <td>   1026.</td>
</tr>
<tr>
  <th>Df Model:</th>               <td>     1</td>       <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>       <td>nonrobust</td>     <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   22.1978</td> <td>    6.335</td> <td>    3.504</td> <td> 0.001</td> <td>    9.607</td> <td>   34.789</td>
</tr>
<tr>
  <th>ones</th>      <td>   22.1978</td> <td>    6.335</td> <td>    3.504</td> <td> 0.001</td> <td>    9.607</td> <td>   34.789</td>
</tr>
<tr>
  <th>Budget</th>    <td>    0.7846</td> <td>    0.133</td> <td>    5.901</td> <td> 0.000</td> <td>    0.520</td> <td>    1.049</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>39.749</td> <th>  Durbin-Watson:     </th> <td>   0.674</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  99.441</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.587</td> <th>  Prob(JB):          </th> <td>2.55e-22</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.091</td> <th>  Cond. No.          </th> <td>4.53e+17</td>
</tr>
</table>



For one unit increase in Budget, the predicted increase in DomesticTotalGross is 0.7846.


```python
plt.plot(fit.predict(X), y, marker = '.', ls = 'None')
plt.title('Model Prediction Against Feature Variable')
plt.xlabel('Budget')
plt.ylabel('Domestic Total Gross');
```


![png](/images/TrainTestSplit_files/TrainTestSplit_16_0.png)



```python
plt.plot(fit.resid, y, marker = '.', ls = 'None')
plt.title('Residuals Against Feature Variable')
plt.xlabel('Budget')
plt.ylabel('Residual');
```


![png](/images/TrainTestSplit_files/TrainTestSplit_17_0.png)


### Challenge 3
Repeat the process of challenge 1, but add a categorical feature (like genre). You'll have to convert a column of text into a number of numerical columns ("dummy variables"). How can you interpret what's happening in the model?


```python
df = df.join(pd.get_dummies(df.Rating))
df.drop(['Rating', 'PG-13'], axis = 1, inplace = True)
df.head().round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Budget</th>
      <th>Domestic Total Gross</th>
      <th>Director</th>
      <th>Run Time</th>
      <th>ReleaseDate</th>
      <th>ones</th>
      <th>G</th>
      <th>PG</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Hunger Games: Catching Fire</td>
      <td>130.0</td>
      <td>424.7</td>
      <td>Francis Lawrence</td>
      <td>146</td>
      <td>2013-11-22 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iron Man 3</td>
      <td>200.0</td>
      <td>409.0</td>
      <td>Shane Black</td>
      <td>129</td>
      <td>2013-05-03 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>150.0</td>
      <td>400.7</td>
      <td>Chris BuckJennifer Lee</td>
      <td>108</td>
      <td>2013-11-22 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Despicable Me 2</td>
      <td>76.0</td>
      <td>368.1</td>
      <td>Pierre CoffinChris Renaud</td>
      <td>98</td>
      <td>2013-07-03 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Man of Steel</td>
      <td>225.0</td>
      <td>291.0</td>
      <td>Zack Snyder</td>
      <td>143</td>
      <td>2013-06-14 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y, X = patsy.dmatrices('DomesticTotalGross ~ ones + G + PG + R', data = df, return_type = "dataframe")
model = sm.OLS(y, X)
fit = model.fit()
fit.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>DomesticTotalGross</td> <th>  R-squared:         </th> <td>   0.109</td>
</tr>
<tr>
  <th>Model:</th>                    <td>OLS</td>        <th>  Adj. R-squared:    </th> <td>   0.081</td>
</tr>
<tr>
  <th>Method:</th>              <td>Least Squares</td>   <th>  F-statistic:       </th> <td>   3.924</td>
</tr>
<tr>
  <th>Date:</th>              <td>Mon, 19 Feb 2018</td>  <th>  Prob (F-statistic):</th>  <td>0.0109</td>
</tr>
<tr>
  <th>Time:</th>                  <td>14:37:46</td>      <th>  Log-Likelihood:    </th> <td> -582.65</td>
</tr>
<tr>
  <th>No. Observations:</th>       <td>   100</td>       <th>  AIC:               </th> <td>   1173.</td>
</tr>
<tr>
  <th>Df Residuals:</th>           <td>    96</td>       <th>  BIC:               </th> <td>   1184.</td>
</tr>
<tr>
  <th>Df Model:</th>               <td>     3</td>       <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>       <td>nonrobust</td>     <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   55.7249</td> <td>    6.109</td> <td>    9.122</td> <td> 0.000</td> <td>   43.598</td> <td>   67.851</td>
</tr>
<tr>
  <th>ones</th>      <td>   55.7249</td> <td>    6.109</td> <td>    9.122</td> <td> 0.000</td> <td>   43.598</td> <td>   67.851</td>
</tr>
<tr>
  <th>G</th>         <td>  157.0430</td> <td>   84.651</td> <td>    1.855</td> <td> 0.067</td> <td>  -10.987</td> <td>  325.073</td>
</tr>
<tr>
  <th>PG</th>        <td>   19.6859</td> <td>   24.840</td> <td>    0.792</td> <td> 0.430</td> <td>  -29.622</td> <td>   68.994</td>
</tr>
<tr>
  <th>R</th>         <td>  -41.5573</td> <td>   18.410</td> <td>   -2.257</td> <td> 0.026</td> <td>  -78.100</td> <td>   -5.014</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>46.897</td> <th>  Durbin-Watson:     </th> <td>   0.240</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 109.261</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.826</td> <th>  Prob(JB):          </th> <td>1.88e-24</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.589</td> <th>  Cond. No.          </th> <td>2.76e+16</td>
</tr>
</table>



According to the model, G-rated movie is predicted to earn 1.57e08 more than PG-13 movie; PG-rated movie is predicted to earn 1.969e07 more than PG-13 movie; and R-rated movie is expected to make 4.156e07 less than PG-13 movie. The PG-13 movie is predicted to make 5.572e07 Domestic Total Gross.

### Challenge 4
Enhance your model further by adding more features and/or transforming existing features. Think about how you build the model matrix and how to interpret what the model is doing.


```python
df.head().round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Budget</th>
      <th>Domestic Total Gross</th>
      <th>Director</th>
      <th>Run Time</th>
      <th>ReleaseDate</th>
      <th>ones</th>
      <th>G</th>
      <th>PG</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Hunger Games: Catching Fire</td>
      <td>130.0</td>
      <td>424.7</td>
      <td>Francis Lawrence</td>
      <td>146</td>
      <td>2013-11-22 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iron Man 3</td>
      <td>200.0</td>
      <td>409.0</td>
      <td>Shane Black</td>
      <td>129</td>
      <td>2013-05-03 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>150.0</td>
      <td>400.7</td>
      <td>Chris BuckJennifer Lee</td>
      <td>108</td>
      <td>2013-11-22 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Despicable Me 2</td>
      <td>76.0</td>
      <td>368.1</td>
      <td>Pierre CoffinChris Renaud</td>
      <td>98</td>
      <td>2013-07-03 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Man of Steel</td>
      <td>225.0</td>
      <td>291.0</td>
      <td>Zack Snyder</td>
      <td>143</td>
      <td>2013-06-14 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y, X = patsy.dmatrices('DomesticTotalGross ~ ones + G + PG + R + Runtime', data = df, return_type = "dataframe")
model = sm.OLS(y, X)
fit = model.fit()
fit.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>DomesticTotalGross</td> <th>  R-squared:         </th> <td>   0.215</td>
</tr>
<tr>
  <th>Model:</th>                    <td>OLS</td>        <th>  Adj. R-squared:    </th> <td>   0.182</td>
</tr>
<tr>
  <th>Method:</th>              <td>Least Squares</td>   <th>  F-statistic:       </th> <td>   6.497</td>
</tr>
<tr>
  <th>Date:</th>              <td>Mon, 19 Feb 2018</td>  <th>  Prob (F-statistic):</th> <td>0.000115</td>
</tr>
<tr>
  <th>Time:</th>                  <td>14:37:46</td>      <th>  Log-Likelihood:    </th> <td> -576.35</td>
</tr>
<tr>
  <th>No. Observations:</th>       <td>   100</td>       <th>  AIC:               </th> <td>   1163.</td>
</tr>
<tr>
  <th>Df Residuals:</th>           <td>    95</td>       <th>  BIC:               </th> <td>   1176.</td>
</tr>
<tr>
  <th>Df Model:</th>               <td>     4</td>       <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>       <td>nonrobust</td>     <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  -41.6434</td> <td>   27.847</td> <td>   -1.495</td> <td> 0.138</td> <td>  -96.927</td> <td>   13.640</td>
</tr>
<tr>
  <th>ones</th>      <td>  -41.6434</td> <td>   27.847</td> <td>   -1.495</td> <td> 0.138</td> <td>  -96.927</td> <td>   13.640</td>
</tr>
<tr>
  <th>G</th>         <td>  174.4610</td> <td>   80.042</td> <td>    2.180</td> <td> 0.032</td> <td>   15.558</td> <td>  333.364</td>
</tr>
<tr>
  <th>PG</th>        <td>   48.8148</td> <td>   24.821</td> <td>    1.967</td> <td> 0.052</td> <td>   -0.461</td> <td>   98.090</td>
</tr>
<tr>
  <th>R</th>         <td>  -30.3201</td> <td>   17.657</td> <td>   -1.717</td> <td> 0.089</td> <td>  -65.374</td> <td>    4.734</td>
</tr>
<tr>
  <th>Run Time</th>   <td>    1.6572</td> <td>    0.464</td> <td>    3.574</td> <td> 0.001</td> <td>    0.737</td> <td>    2.578</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>43.608</td> <th>  Durbin-Watson:     </th> <td>   0.448</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  95.295</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.721</td> <th>  Prob(JB):          </th> <td>2.03e-21</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.319</td> <th>  Cond. No.          </th> <td>5.97e+18</td>
</tr>
</table>



### Challenge 5
Fitting and checking predictions on the exact same data set can be misleading. Divide your data into two sets: a training and a test set (roughly 75% training, 25% test is a fine split). Fit a model on the training set, check the predictions (by plotting versus actual values) in the test set.


```python
df.dropna(inplace = True)
y = df.DomesticTotalGross
X = df.drop(['DomesticTotalGross', 'Title', 'Director', 'ReleaseDate', 'G'], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
```


```python
lr = LinearRegression()
lr.fit(X_train, y_train)
plt.plot(lr.predict(X_test), y_test, marker = '.', ls = 'None')
plt.title('Predictions versus Actual Values')
plt.xlabel('Predicted')
plt.ylabel('Actual');
```


![png](/images/TrainTestSplit_files/TrainTestSplit_27_0.png)
