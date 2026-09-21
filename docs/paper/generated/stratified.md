# 分组配对增益（pp）

稀疏组不是零历史冷启动；桶边界来自训练频次，全部非空桶都报告。

| Dataset | Group | N | ΔR1 | ΔR5 | ΔR10 | ΔN10 | ΔMRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nyc | train_user_frequency_Q1 | 218 | 3.211 | -0.459 | 2.294 | 2.442 | 2.389 |
| nyc | train_user_frequency_Q2 | 213 | 2.347 | -0.939 | 0.939 | 1.591 | 1.776 |
| nyc | train_user_frequency_Q3 | 235 | 1.702 | 0.426 | 0.000 | 0.577 | 0.801 |
| nyc | train_user_frequency_Q4 | 681 | 1.762 | 2.203 | 1.175 | 1.545 | 1.533 |
| nyc | candidate_categories_2_4 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| nyc | candidate_categories_5_9 | 101 | 1.980 | 0.990 | 0.000 | 1.007 | 1.382 |
| nyc | candidate_categories_10_20 | 1245 | 2.088 | 0.964 | 1.205 | 1.572 | 1.599 |
| nyc | seen_user_POI_in_train | 971 | 2.987 | 1.648 | 1.339 | 2.164 | 2.315 |
| nyc | unseen_user_POI_in_train | 376 | -0.266 | -0.798 | 0.532 | -0.110 | -0.310 |
| ca | train_user_frequency_Q1 | 508 | 0.787 | 0.984 | 0.984 | 0.751 | 0.593 |
| ca | train_user_frequency_Q2 | 324 | 0.000 | 2.778 | 0.926 | 0.743 | 0.590 |
| ca | train_user_frequency_Q3 | 505 | 1.980 | 0.792 | 0.198 | 1.263 | 1.596 |
| ca | train_user_frequency_Q4 | 1443 | 0.624 | 0.139 | -0.139 | 0.343 | 0.491 |
| ca | candidate_categories_1_1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ca | candidate_categories_2_4 | 37 | 0.000 | 0.000 | 0.000 | -0.118 | -0.168 |
| ca | candidate_categories_5_9 | 183 | -0.546 | 1.093 | 2.732 | 0.870 | 0.081 |
| ca | candidate_categories_10_20 | 2559 | 0.938 | 0.703 | 0.078 | 0.625 | 0.781 |
| ca | seen_user_POI_in_train | 1346 | 1.932 | 1.337 | 1.040 | 1.608 | 1.724 |
| ca | unseen_user_POI_in_train | 1434 | -0.209 | 0.139 | -0.488 | -0.285 | -0.219 |
| tky | train_user_frequency_Q1 | 1207 | 0.580 | 0.746 | 0.331 | 0.444 | 0.439 |
| tky | train_user_frequency_Q2 | 1496 | 0.535 | 0.334 | 0.134 | 0.325 | 0.372 |
| tky | train_user_frequency_Q3 | 1627 | 0.123 | 0.738 | 0.369 | 0.450 | 0.417 |
| tky | train_user_frequency_Q4 | 2708 | 0.258 | 0.258 | 0.222 | 0.232 | 0.208 |
| tky | candidate_categories_1_1 | 24 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| tky | candidate_categories_2_4 | 763 | 0.131 | -0.262 | 0.262 | 0.137 | 0.076 |
| tky | candidate_categories_5_9 | 3062 | 0.163 | 0.000 | 0.229 | 0.233 | 0.203 |
| tky | candidate_categories_10_20 | 3189 | 0.564 | 1.098 | 0.282 | 0.490 | 0.517 |
| tky | seen_user_POI_in_train | 4845 | 0.537 | 0.784 | 0.372 | 0.535 | 0.543 |
| tky | unseen_user_POI_in_train | 2193 | -0.091 | -0.228 | 0.000 | -0.095 | -0.137 |
