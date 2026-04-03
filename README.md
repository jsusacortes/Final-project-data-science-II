# Final-project-data-science-II

## Source and Link
Dataset: Bank Customer Churn Modelling — available on Kaggle at
https://www.kaggle.com/datasets/shubh0799/churn-modelling

## Dataset Overview
This dataset contains 10,000 observations and 14 features. Each observation
represents a single bank customer's relationship snapshot, capturing demographic
information, account behavior, and product usage. The dataset is used to predict
whether a customer will exit (churn) the bank.

## Verification of Requirements
This dataset satisfies the project requirements: it contains 10,000 observations
(exceeding the 1,000+ threshold) and 14 features (exceeding the 10+ threshold). After
excluding identifiers (RowNumber, CustomerId, Surname), 11 meaningful features
remain for modeling.

## Results - summary

**Feature Engineering Plans:**
I currently have 3 engineered features (BalanceSalaryRatio, AgeGenderInteraction, 
HasZeroBalance). For the final submission I plan to add a tenure group (binned 
NumOfProducts) and a products-per-active-member ratio, bringing the total to 5+.

**Model Optimization Plans:**
Both models achieved a ROC-AUC of ~0.86, which is a solid baseline before any 
tuning. However, accuracy and F1 scores leave room for improvement. For Random 
Forest, I will tune n_estimators, max_depth, and min_samples_split using 
GridSearchCV to improve precision and F1. For the Neural Network, I will 
experiment with additional layers, learning rate scheduling, and different 
dropout rates since it currently has better Recall (0.715 vs 0.629) but takes 
15x longer to train (21.69s vs 1.42s). The key trade-off is that Random Forest 
is faster and more precise while the Neural Network catches more actual churners, 
which matters most from a business perspective since missing a churner is more 
costly than a false alarm.
