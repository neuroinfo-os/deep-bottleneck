**********************
Description of cohorts
**********************

The experiments are structured in different cohort, containing one specific variation of parameters.
To show the aim of the cohorts and to simplify the access of the saved artifacts using the artifact-viewer
the following table offers a simple description for each cohort. 


+------------+-----------------------------------------------------------------------------------+
| Cohort     | Description                                                                       |
+============+===================================================================================+
| cohort_1   | Comparison of upper, lower and binning as different estimator.                    |
|            | Additionally the hyperparameter of the estimators are varied.                     |
|            | All experiments are done for relu and tanh using sgd as optimizer.                |
+------------+-----------------------------------------------------------------------------------+
| cohort_2   | Comparison of training-, test- and full-dataset as base for the mi-computation.   |
|            | All experiments are done for relu and tanh using sgd as optimizer.                |
+------------+-----------------------------------------------------------------------------------+
| cohort_3   | Comparison of upper, lower and binning as different estimator.                    |
|            | Additionally the hyperparameter of the estimators are varied.                     |
|            | All experiments are done for relu and tanh using adam as optimizer.               |
+------------+-----------------------------------------------------------------------------------+
| cohort_4   | Comparison of training-, test- and full-dataset as base for the mi-computation.   |
|            | All experiments are done for relu and tanh using Adam as optimizer.               |
+------------+-----------------------------------------------------------------------------------+
| cohort_5   | Comparison of different standard activation functions.                            |
|            | All experiments are done using adam as optimizer.                                 |
+------------+-----------------------------------------------------------------------------------+
| cohort_6   | Comparison of basic architectures.                                                |
|            | All experiments are done for relu and tanh using adam as optimizer.               |
+------------+-----------------------------------------------------------------------------------+
| cohort_7   | Comparison of different hyperparameter for max-norm regularization.               |
|            | All experiments are done for relu and tanh using adam as optimizer.               |
+------------+-----------------------------------------------------------------------------------+