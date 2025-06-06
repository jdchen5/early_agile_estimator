# Summary of 2_Default_LightGBM

[<< Go back](../README.md)


## LightGBM
- **n_jobs**: -1
- **objective**: regression
- **num_leaves**: 63
- **learning_rate**: 0.05
- **feature_fraction**: 0.9
- **bagging_fraction**: 0.9
- **min_data_in_leaf**: 10
- **metric**: rmse
- **custom_eval_metric_name**: None
- **explain_level**: 2

## Validation
 - **validation_type**: kfold
 - **k_folds**: 10

## Optimized metric
rmse

## Training time

106.8 seconds

### Metric details:
| Metric   |          Score |
|:---------|---------------:|
| MAE      |  715.643       |
| MSE      |    2.19632e+07 |
| RMSE     | 4686.49        |
| R2       |    0.783667    |
| MAPE     |    0.135748    |



## Learning curves
![Learning curves](learning_curves.png)

## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
