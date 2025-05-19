# Summary of 14_Xgboost

[<< Go back](../README.md)


## Extreme Gradient Boosting (Xgboost)
- **n_jobs**: -1
- **objective**: reg:squarederror
- **eta**: 0.05
- **max_depth**: 4
- **min_child_weight**: 5
- **subsample**: 0.6
- **colsample_bytree**: 0.5
- **eval_metric**: rmse
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.8

## Optimized metric
rmse

## Training time

4.1 seconds

### Metric details:
| Metric   |          Score |
|:---------|---------------:|
| MAE      | 2309.71        |
| MSE      |    6.42305e+07 |
| RMSE     | 8014.39        |
| R2       |    0.47681     |
| MAPE     |    0.584241    |



## Learning curves
![Learning curves](learning_curves.png)

## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
