---
layout: post
title: Regression Metrics
---


### $$R^2$$ Score

The $$R^2$$ statistics is the amount of variance in the dependent variable explained by your model. It is given by the formula:

$$R^2 = \frac{ESS}{TSS} = 1 − \frac{RSS}{TSS}$$

where

$$ESS = \sum\limits_{i=1}^n (\hat y_i - \bar y_i)^2, \
RSS = \sum\limits_{i=1}^n (y_i - \hat y_i)^2, \ and \ \  
TSS = \sum\limits_{i=1}^n (y_i - \bar y_i)^2 \\
where \ \  \bar y_i = \sum\limits_{i=1}^n y_i$$

The explained sum of squares ($$ESS$$) is a quantity used in describing how well a model represents the data being modeled. In particular, the explained sum of squares measures how much variation there is in the modeled values.

The residual sum of squares ($$RSS$$) is the sum of the squares of residuals. The residuals are deviations of predicted from the actual empirical values of data. $$RSS$$ is a measure of the discrepancy between the data and an estimation model. A small $$RSS$$ indicates a tight fit of the model to the data. The residual sum of squares measures the variation in the modeling errors.

The total sum of squares ($$TSS$$) is the sum of the squares of the difference of the dependent variable and its mean. The total sum of squares measures how much variation there is in the observed data.

In general, total sum of squares = explained sum of squares + residual sum of squares
($$TSS = ESS + RSS$$).

### Mean Squared Error and Root-Mean-Square Error

The mean squared error ($$MSE$$) is another popular metrics in regression settings. The mean squared error is the residual sum of squares divided by the number of observations.

$$MSE = \frac{1}{n} \cdot RSS = \frac{1}{n} \sum\limits_{i=1}^n (y_i - \hat y_i)^2$$

The root-mean-square error ($$RMSE$$) is the square root of $$MSE$$ (i.e. $$RMSE = \sqrt{MSE}$$). $$RMSE$$ has an advantage over $$MSE$$ because it has the same units as the quantity being estimated.

$$R^2$$ is a standardized measure (from 0 to 1) of model fit. $$MSE$$ is the estimate of variance of residuals, or non-fit. We want higher $$R^2$$ and lower $$MSE$$. The two measures are related as can be seen from here:

$$R^2 = 1 − \frac{RSS}{TSS} = 1 - \frac{n \cdot MSE}{TSS}$$
