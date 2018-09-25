# Katabat Machine Learning pipeline:
##### Note: pipeline code is in the 'automl' folder. Other folders are for the experimental purpose.

![alt text](https://github.com/wluo-personal/AlphaBoosting/blob/master/pipeline.PNG)
* Stage 1: Feature Engineering: make the data ready for machine learning.
  * Data imputation and outlier handling ^
  * Categorical features encoding by Cat2Vec or Deep-learning Embedding.
  * Time-series historical data aggregation using Recurrent Neural Network. *
  * Automated feature engineering for relational datasets. *
* Stage 2: Feature Selection: refine features by correlation and importance analysis to reduce noises.
* Stage 3: Model Exploring: build a variety of models including:
  * Gradient Boosting Machines (LightGBM, CatBoost, Xgboost), 
  * Neural Network
  * Factorization Machines and Collaborative Filtering 
  * Traditional models (Logistic/linear Regression, SVM, Random Forest)
* Stage 4: Model Tuning:  using Bayesian Optimization to set up a guided search for the best hyperparameters. *
* Stage 5: Model Ensembling: construct a StackNet to find the optimal combination of diverse models for better performance.
* Stage 6: Model Explanation: utilize a unified approach (SHAP) to explain the output of any machine learning model. 
##### * on-going   ^ manual


