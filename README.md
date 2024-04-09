# Data Analysis and Prediction of Rainfall

## Introduction

This repository contains an analysis of rainfall data and attempts to predict rainfall using various machine learning algorithms. The analysis includes data preprocessing, exploration, and model evaluation.
Data Analysis

Upon initial inspection of the data, several issues were identified and addressed:

   * Some columns contain missing data, notably 'Evaporation' and 'Sunshine'.
   * The 'Wind Speed' column contains non-numerical values such as 'Calm', which were converted to numerical values (0).
   * Missing values in the 'Rainfall' column were addressed by populating them with 0.
   * Utilization of daily temperature measurements instead of those recorded at specific times.
   * Aggregation of '9am/3pm relative humidity' measurements into 'Median daily humidity'.
   * Usage of 'Speed of maximum wind gust' measurements instead of individual wind speeds, with missing values filled with the maximum of '9am/3pm wind speed'.
   * The features of the dataset are at different scales so, normalizing it before training will help us to obtain optimum results faster along with stable training.


### Feature Selection

Based on the analysis, the following features were selected for predicting rainfall:

    Minimum temperature
    Maximum temperature
    Median humidity
    Speed of maximum wind gust

## Prediction and Model Evaluation

### Linear Regression

There is no linear relationship between the independent and depended (Rainfall) variables. 

![](images/scatter.png)

Linear regression was initially attempted but did not yield accurate results:

    Coefficients: [[-0.55605176 -1.09944499 1.00342415 2.15911104]]
    Intercept: [4.88449198]
    Mean Absolute Error: 4.92
    Mean Squared Error: 49.47
    Variance Score (R2-score): 0.24

### K Neighbors Regressor

K Neighbors Regressor was also tried, but the results were not satisfactory:

    k = 4: MAE=4.20, MSE=41.33, R2-score=0.37
    k = 5: MAE=4.06, MSE=40.52, R2-score=0.38
    k = 6: MAE=4.12, MSE=39.73, R2-score=0.39

### Binary Classification

![](images/bar.png)

To improve prediction, binary classification was attempted to predict whether it will rain or not. Various classifiers were evaluated:
~~~
| Algorithm            | Precision | Recall   | F1-score | Accuracy |
|----------------------|-----------|----------|----------|----------|
| KNN                  | 0.774     | 0.815    | 0.776    | 0.815    |
| Decision Trees       | 0.681     | 0.723    | 0.700    | 0.723    |
| Logistic Regression  | 0.871     | 0.846    | 0.798    | 0.846    |
| SVM                  | 0.665     | 0.815    | 0.732    | 0.815    |
~~~

### Conclusion

While logistic regression performed relatively better compared to other models, further improvements may be necessary for more accurate predictions.

In conclusion, despite attempts with various machine learning algorithms, accurately predicting rainfall remains challenging. Further refinement of models and feature engineering may be required for better results.

### Contributor

  Masha Orfali (masha.orfali@gmail.com)


### References

The raw data for Daily Weather Observations in Luncheon Hill, Tasmania can be found here:  
 [Australian Weather](http://www.bom.gov.au/climate/dwo/202310/html/IDCJDW7030.202310.shtml)


