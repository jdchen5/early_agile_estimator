# Summary of 1_Linear

[<< Go back](../README.md)


## Linear Regression (Linear)
- **n_jobs**: -1
- **explain_level**: 2

## Validation
 - **validation_type**: kfold
 - **k_folds**: 10

## Optimized metric
rmse

## Training time

74.0 seconds

### Metric details:
| Metric   |            Score |
|:---------|-----------------:|
| MAE      | 664872           |
| MSE      |      2.51488e+15 |
| RMSE     |      5.01486e+07 |
| R2       |     -2.47712e+07 |
| MAPE     |      4.14239     |



## Learning curves
![Learning curves](learning_curves.png)

## Coefficients
| feature                                                                     |    Learner_1 |    Learner_2 |    Learner_3 |    Learner_4 |    Learner_5 |    Learner_6 |    Learner_7 |    Learner_8 |    Learner_9 |   Learner_10 |
|:----------------------------------------------------------------------------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| tech_tf_development_platform_proprietary                                    | -0.0644431   | -0.0648285   | -0.0695124   | -0.0632339   |  5.25508e+11 | -0.0645139   | -0.0692252   | -0.0714662   | -0.0704463   | -0.0583124   |
| project_prf_normalised_work_effort_level_1                                  |  0.315131    |  0.336735    |  0.31694     |  0.330189    |  0.327342    |  0.322013    |  0.31806     |  0.321258    |  0.316837    |  0.323548    |
| project_prf_max_team_size                                                   |  0.206218    |  0.190617    |  0.210623    |  0.192042    |  0.205958    |  0.199665    |  0.205085    |  0.202455    |  0.200997    |  0.205785    |
| project_prf_functional_size                                                 |  0.180966    |  0.158752    |  0.171198    |  0.168353    |  0.168488    |  0.174878    |  0.174397    |  0.173558    |  0.174154    |  0.172872    |
| tech_tf_language_type_3GL                                                   |  0.0948482   |  0.148583    |  0.125543    |  0.143619    |  0.125476    |  0.152357    |  0.141249    |  0.0927904   |  0.129945    |  0.128826    |
| tech_tf_language_type_Missing                                               |  0.0838959   |  0.13785     |  0.104108    |  0.120482    |  0.109749    |  0.124501    |  0.112292    |  0.0781692   |  0.109988    |  0.100555    |
| external_eef_organisation_type                                              |  0.0974419   |  0.0941955   |  0.098474    |  0.090532    |  0.0963864   |  0.096162    |  0.0935893   |  0.0986865   |  0.0902135   |  0.0907482   |
| project_prf_development_type_new_development                                |  0.089442    |  0.0857254   |  0.0860275   |  0.0874679   |  0.0895348   |  0.0843961   |  0.0875918   |  0.0910329   |  0.0896247   |  0.0897318   |
| tech_tf_development_platform_mf                                             |  0.0661326   |  0.0857535   |  0.0798951   |  0.0796023   |  0.079745    |  0.0795658   |  0.0786646   |  0.0777348   |  0.0777892   |  0.0767644   |
| project_prf_application_type                                                |  0.0625281   |  0.0689804   |  0.0671312   |  0.0641946   |  0.0661378   |  0.0643009   |  0.0630718   |  0.0599011   |  0.0674622   |  0.0619492   |
| _case_tool_used_no                                                          |  0.058188    |  0.0647058   |  0.0604619   |  0.0664492   |  0.0543018   |  0.0570682   |  0.0584541   |  0.0672551   |  0.0576209   |  0.0595335   |
| tech_tf_architecture_yes                                                    |  0.0563628   |  0.0579766   |  0.058438    |  0.044302    |  0.0537114   |  0.0642175   |  0.0643059   |  0.0508609   |  0.0522224   |  0.0579273   |
| tech_tf_language_type_4GL                                                   |  0.0177484   |  0.0735945   |  0.0503835   |  0.0719931   |  0.0488982   |  0.0739859   |  0.0627936   |  0.0290704   |  0.0482846   |  0.0601402   |
| project_prf_application_group_real_time_application                         |  0.048543    |  0.0444802   |  0.0455105   |  0.0465781   |  0.0486666   |  0.0442359   |  0.0418549   |  0.0493622   |  0.0470218   |  0.0489126   |
| tech_tf_architecture_don_t_know                                             |  0.0473812   |  0.0442707   |  0.0429533   |  0.0403564   |  0.043299    |  0.0439271   |  0.043547    |  0.0442418   |  0.0393042   |  0.0422801   |
| project_prf_development_type_re_development                                 |  0.0419736   |  0.0407408   |  0.0396909   |  0.0387861   |  0.041826    |  0.0426093   |  0.038302    |  0.0417428   |  0.0407629   |  0.0390851   |
| project_prf_team_size_group                                                 |  0.0401604   |  0.0367045   |  0.0426093   |  0.0384773   |  0.0413108   |  0.0318865   |  0.0366708   |  0.0457582   |  0.0443744   |  0.0381188   |
| tech_tf_language_type_APG                                                   |  0.0282318   |  0.0467674   |  0.0353452   |  0.0448084   |  0.0335922   |  0.0427824   |  0.0464341   |  0.0293152   |  0.0328352   |  0.0382475   |
| project_prf_year_of_project                                                 |  0.0314527   |  0.0404095   |  0.0303643   |  0.033868    |  0.0375644   |  0.0319578   |  0.043077    |  0.0437141   |  0.0447584   |  0.0386588   |
| _case_tool_used_yes                                                         |  0.0340176   |  0.0404757   |  0.0397556   |  0.0370904   |  0.0342281   |  0.0365989   |  0.037373    |  0.0446543   |  0.0372148   |  0.0322614   |
| tech_tf_development_platform_mr                                             |  0.0314272   |  0.0331398   |  0.0306811   |  0.0344278   |  0.0277072   |  0.0289471   |  0.0328831   |  0.0263093   |  0.0293388   |  0.0311567   |
| project_prf_application_group_mathematically_intensive_application_1        |  0.029391    |  0.0275768   |  0.0241333   |  0.0291124   |  0.0282083   |  0.0271017   |  0.0263714   |  0.0267104   |  0.0241867   |  0.0327722   |
| tech_tf_dbms_used_no                                                        |  0.0251041   |  0.0187721   |  0.0236573   |  0.0258831   |  0.0247665   |  0.0246634   |  0.025359    |  0.0221802   |  0.0304853   |  0.0220784   |
| process_pmf_docs                                                            |  0.022219    |  0.0165185   |  0.0203395   |  0.0200153   |  0.0199789   |  0.0170893   |  0.0173764   |  0.019109    |  0.0197887   |  0.0180873   |
| tech_tf_development_platform_multi                                          |  0.00504262  |  0.0149191   |  0.00452308  |  0.0121458   |  0.00491949  |  0.00929893  |  0.000857565 |  0.0026678   | -0.00322991  |  0.000600664 |
| tech_tf_primary_programming_language                                        |  0.00185438  |  0.00751552  | -0.0011683   |  0.00125989  |  0.0118708   | -0.000656518 |  0.000407384 | -0.00190549  |  0.00102446  |  0.00431314  |
| tech_tf_architecture_no                                                     |  0.0089366   |  0.000796162 |  0.00227439  |  0.0011204   |  0.000453339 |  0.00486523  |  0.00888197  | -0.00259709  | -0.0014704   | -3.44477e-05 |
| _case_tool_used_don_t_know                                                  | -0.00422303  | -0.00265693  |  0.00274398  |  0.00364758  | -0.0031897   | -0.00389422  |  0.00140208  |  0.00384693  | -0.00152203  |  0.00604976  |
| intercept                                                                   | -1.92206e-16 | -3.86032e-16 | -3.62503e-16 |  3.08472e-16 |  7.8605e-16  | -1.28703e-16 |  4.32267e-16 |  3.65435e-16 | -5.17521e-16 | -1.02427e-16 |
| project_prf_application_group_business_application__infrastructure_software | -0.00282349  | -0.00289263  | -0.00259791  | -1.94289e-16 | -0.00243051  | -0.00275525  | -0.00275251  | -0.00262181  | -0.00231147  | -0.00261562  |
| project_prf_application_group_mathematically_intensive_application          | -0.00395465  | -0.00352526  | -0.00397139  | -0.00270597  | -0.00370342  | -0.00326465  | -0.00403005  | -0.00239056  | -0.0034761   | -0.0031358   |
| tech_tf_development_platform_hand_held                                      | -0.00501664  | -0.00447177  | -3.55618e-17 | -0.00464035  | -0.0047087   | -0.00433549  | -0.00422568  | -0.00555186  | -0.00474984  | -0.00447923  |
| tech_tf_architecture_not_applicable                                         | -0.00411679  | -0.00685352  | -0.00479875  | -0.00749853  |  0.00160793  | -0.00856653  | -0.00441451  | -0.00534665  | -0.00483079  | -0.00561265  |
| tech_tf_tools_used                                                          | -0.0127011   | -0.00979455  | -0.00707751  | -0.00116068  | -0.0125838   | -0.00934324  | -0.0080212   | -0.00943388  | -0.0123971   | -0.00819385  |
| project_prf_application_group_infrastructure_software                       | -0.019085    | -0.0269591   | -0.0207608   | -0.0176369   | -0.0203523   | -0.0278699   | -0.0210767   | -0.0209964   | -0.0221857   | -0.0204079   |
| external_eef_data_quality_rating_b                                          | -0.0250085   | -0.0243832   | -0.0278908   | -0.0190397   | -0.0263683   | -0.0203783   | -0.0227949   | -0.0282407   | -0.031883    | -0.0233449   |
| tech_tf_dbms_used_yes                                                       | -0.0251081   | -0.0344261   | -0.0344745   | -0.0314448   | -0.0289965   | -0.0320766   | -0.0263072   | -0.0316123   | -0.0205742   | -0.0306468   |
| tech_tf_development_platform_pc                                             | -0.0421275   | -0.0241276   | -0.0365638   | -0.0318102   | -0.0279058   | -0.029785    | -0.0308771   | -0.0385124   | -0.0362223   | -0.0385365   |
| project_prf_application_group_business_application                          | -0.0570931   | -0.05712     | -0.0592643   | -0.0555688   | -0.0510175   | -0.0528975   | -0.056736    | -0.0533727   | -0.0550783   | -0.0455179   |
| external_eef_industry_sector                                                | -0.0623942   | -0.0631057   | -0.0637965   | -0.0503341   | -0.0605186   | -0.0607818   | -0.0630137   | -0.0614582   | -0.0595492   | -0.0535965   |
| process_pmf_development_methodologies                                       | -0.130187    | -0.141047    | -0.124107    | -0.132143    | -0.138096    | -0.131269    | -0.127173    | -0.137313    | -0.131885    | -0.13191     |
| project_prf_relative_size                                                   | -0.384136    | -0.38376     | -0.389276    | -0.377537    | -0.391585    | -0.386208    | -0.388042    | -0.384464    | -0.386675    | -0.380426    |
| tech_tf_language_type_5GL                                                   |  0.0140173   |  0.0235065   |  0.018944    |  0.0197321   | -5.25508e+11 |  0.0207244   |  0.0209889   |  0.0177306   |  0.0202714   |  0.0162981   |


## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



## SHAP Importance
![SHAP Importance](shap_importance.png)

## SHAP Dependence plots

### Dependence (Fold 1)
![SHAP Dependence from Fold 1](learner_fold_0_shap_dependence.png)
### Dependence (Fold 2)
![SHAP Dependence from Fold 2](learner_fold_1_shap_dependence.png)
### Dependence (Fold 3)
![SHAP Dependence from Fold 3](learner_fold_2_shap_dependence.png)
### Dependence (Fold 4)
![SHAP Dependence from Fold 4](learner_fold_3_shap_dependence.png)
### Dependence (Fold 5)
![SHAP Dependence from Fold 5](learner_fold_4_shap_dependence.png)
### Dependence (Fold 6)
![SHAP Dependence from Fold 6](learner_fold_5_shap_dependence.png)
### Dependence (Fold 7)
![SHAP Dependence from Fold 7](learner_fold_6_shap_dependence.png)
### Dependence (Fold 8)
![SHAP Dependence from Fold 8](learner_fold_7_shap_dependence.png)
### Dependence (Fold 9)
![SHAP Dependence from Fold 9](learner_fold_8_shap_dependence.png)
### Dependence (Fold 10)
![SHAP Dependence from Fold 10](learner_fold_9_shap_dependence.png)

## SHAP Decision plots

### Top-10 Worst decisions (Fold 1)
![SHAP worst decisions from fold 1](learner_fold_0_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 2)
![SHAP worst decisions from fold 2](learner_fold_1_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 3)
![SHAP worst decisions from fold 3](learner_fold_2_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 4)
![SHAP worst decisions from fold 4](learner_fold_3_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 5)
![SHAP worst decisions from fold 5](learner_fold_4_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 6)
![SHAP worst decisions from fold 6](learner_fold_5_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 7)
![SHAP worst decisions from fold 7](learner_fold_6_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 8)
![SHAP worst decisions from fold 8](learner_fold_7_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 9)
![SHAP worst decisions from fold 9](learner_fold_8_shap_worst_decisions.png)
### Top-10 Worst decisions (Fold 10)
![SHAP worst decisions from fold 10](learner_fold_9_shap_worst_decisions.png)
### Top-10 Best decisions (Fold 1)
![SHAP best decisions from fold 1](learner_fold_0_shap_best_decisions.png)
### Top-10 Best decisions (Fold 2)
![SHAP best decisions from fold 2](learner_fold_1_shap_best_decisions.png)
### Top-10 Best decisions (Fold 3)
![SHAP best decisions from fold 3](learner_fold_2_shap_best_decisions.png)
### Top-10 Best decisions (Fold 4)
![SHAP best decisions from fold 4](learner_fold_3_shap_best_decisions.png)
### Top-10 Best decisions (Fold 5)
![SHAP best decisions from fold 5](learner_fold_4_shap_best_decisions.png)
### Top-10 Best decisions (Fold 6)
![SHAP best decisions from fold 6](learner_fold_5_shap_best_decisions.png)
### Top-10 Best decisions (Fold 7)
![SHAP best decisions from fold 7](learner_fold_6_shap_best_decisions.png)
### Top-10 Best decisions (Fold 8)
![SHAP best decisions from fold 8](learner_fold_7_shap_best_decisions.png)
### Top-10 Best decisions (Fold 9)
![SHAP best decisions from fold 9](learner_fold_8_shap_best_decisions.png)
### Top-10 Best decisions (Fold 10)
![SHAP best decisions from fold 10](learner_fold_9_shap_best_decisions.png)

[<< Go back](../README.md)
