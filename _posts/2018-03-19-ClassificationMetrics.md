---
layout: post
title: Classification Metrics
---


## Confusion Matrix

<table style="text-align:center;">
  <tr>
  	<td colspan="2" rowspan="2"></td>
    <td colspan="2">Predicted</td>
  </tr>
  <tr>
    <td>Positive</td>
    <td>Negative</td>
  </tr>
  <tr>
    <td rowspan="2">Actual</td>
    <td>Positive</td>
    <td >True Positive</td>
    <td >False Negative <br />
    (Type II error)</td>
  </tr>
  <tr>
    <td>Negative</td>
    <td >False Positive <br />
    (Type I error)</td>
    <td >True Negative</td>
  </tr>
</table>

|                 | Predicted |  Actual  |
|:---------------:|:---------:|:--------:|
| True Positive:  | Positive  | Positive |
| False Negative: | Negative  | Positive |
| False Positive: | Positive  | Negative |
| True Negative:  | Negative  | Negative |

A lot of people (including myself) confuse False Negative and False Positive.  

Tip for memorizing: They are both False because the model got them wrong, but if it is predicted Negative then it is negative and if it is predicted Positive, then it is positive.

## Accuracy Score

Accuracy is the number of correct predictions out of all predictions.  

$$Accuracy = \frac{True Positive + True Negative}{True Positive + True Negative + False Positive + False Negative}$$  

Caveats: if the classes are highly imbalanced, the model with high accuracy might be useless. For example, if the people with cancer is 1% of the total population, the model predicting everybody as not having cancer will achieve 99% accuracy but this is not desirable. Let's take a look at other metrics.

## Precision

Precision is out of all the cases the model predicted Positive, how often is it correct?  

$$Precision = \frac{True Positive}{True Positive + False Positive}$$  

I used precision as my metric when classifying whether customer is going to tip or not. This is because when my model predict Positive to tip, I want the customers to tip. It will look bad if my model says the customer will tip and they end up not tipping.

## Recall (Sensitivity)

Recall, sensitivity or true positive rate (TPR) is when actually Positive, how often does the model predict Positive?  

$$Recall = \frac{True Positive}{True Positive + False Negative}$$  

Going back to the cancer case, we want to really hone in on recall because we want to really identify the people with cancer. It is more acceptable to misclassify some people as having cancer and suggest further tests then to miss the people with actual cancer.

There is a Precision / Recall Tradeoff: the higher the recall, the lower the precision and vice versa.

## $$F_1$$ Score

$$F_1$$ score is the harmonic (weighted) mean of precision and recall. The harmonic mean gives much more weight to low values. As a result, the classifier will only get a high $$F_1$$ score if both recall and precision are high.  

$$F_1 = \frac{2}{\frac{1}{precistion}+{\frac{1}{recall}}}
= 2\cdot\frac{precision\cdot recall}{precision+recall}
= \frac{TP}{TP + \frac{FN+FP}{2}}$$  

$$F_1$$ score gives a convenient way to compare classifiers.

## The ROC Curve

The ROC (Receiver Operating Characteristic) curve plots the true positive rate (recall) against the false positive rate (FPR). The FPR is the ratio of negative instances that are incorrectly classified as positive. Therefore, FPR is equal to one minus the true negative rate (TNR), which is the ratio of negative instances that are correctly classified as negative. Since the TNR is also called specificity, the ROC curve plots sensitivity (recall) versus 1 – specificity.

Once again there is a tradeoff: the higher the recall (TPR), the more false positives (FPR) the classifier produces.

One way to compare classifiers is to measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
