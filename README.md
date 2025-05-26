# Module3Capstone - Documentation
Amanda Sibuea

## Main Topic: Churn Signals - Listening to What Data Says About Leaving Customers
Predicting Customer Churn in E-Commerce Using Machine Learning

## Introduction
This capstone project for the Purwadhika Data Science Programme focuses on exploring data from start to finish regarding E-Commerce Customer Churn.
The goal of this project is to apply the skills and knowledge regarding machine learning gained in Module 3 in a practical case study.
Demonstrating my understanding of key concepts and usage of tools covered in the module.

## Problem Description
The data set belongs to a leading online E-commerce company. An online retail company wants to know the customers who are going to churn, so accordingly, they can approach customers to offer some promos.
With that, the goal of this project is to create a prediction model to accurately predict the customers who will potentially churn based on the data of the features collected in the dataset provided.

## Research Question
Can predictive modeling of churn improve customer retention efforts in online retail?

### Target
0: Does not churn <br>
1: Churns

In this project, a churned customer is defined based on the label provided in the dataset.
The churn column indicates whether a customer is considered churned (1) or not-churned (0), likely based on business rules such as inactivity or purchase patterns.
Although the specific criteria used to assign this label weren’t explicitly provided, the model uses available behavioral features like Day Since Last Order, Tenure, and Complaint history, to learn patterns associated with customer churn.

### Features
- Tenure: tenure of a customer in the company
- WarehouseToHome: distance between the warehouse to the customer's home
- NumberOfDeviceRegistered: total number of devices registered on a particular customer
- PreferedOrderCat: prefered order category of a customer in the last month
- SatisfactionScore: satisfactory score of a customer on service
- MaritalStatus: marital status of a customer
- NumberOfAddress: total number of address on a particular customer
- Complain: whether a customer has raised a complain in the last month
- DaySinceLastOrder: day since last order by customer
- CashbackAmount : average cashback in last month

### Metrics
#### Type 1 error: false positives
Consequence: more customers offered the promos means more financial resources used

#### Type 2 error: false negatives
Consequence: miss out on the actual churners
<br>
With that, since there is no budget contraints regarding offering the potential churners promos, and the fact that the goal is to catch all churners, the error to minimise is false negative. Therefore, the main metric to be used is recall.
<br>
Since the business priority is to minimize churn by catching all potential churners, we prioritised recall over precision.

#### Cost of errors
*Cost of false positives*
If the company incorrectly flag a loyal customer as “about to churn”, they might:

- spend on retention campaigns unnecessarily
- risk giving rewards to customers who would have stayed anyway

*Cost of false negatives*
If the company misses a churning customer, they will loose:

- future revenue
- customer lifetime value
- referral potential

<br>
Going forward, the dataset assumes that this e-commerce retail company has a high customer lifetime value and low retention cost.

### Data Cleaning and EDA
#### Duplicated Data
Dataset shows 671 rows of duplicated data. The duplicates are dropped to ensure that the data used for the machine learning model is accurate and reliable. Dropping duplicates reduces bias.

#### Data Balance
The raw dataset is not balanced. The number of not churned (0) dominates the dataset while the number of churn (1) takes up a smaller percentage of the dataset.
With that the dataset is imbalanced and is taken into consideration going forward.

#### Missing Values
There was a total of 3 columns with missing values. The columns are: Tenure, WarehouseToHome, and DaySinceLastOrder.
The missing values in each column take up a small percentage of the whole dataset. Before dropping or imputing data to handle the missing values, I first check whether the missing values have any pattern or relationship between each other.
<br>
The matrix plot shows how the missing values in each of the three collumns does not have a pattern.
To support the claim that the missing values have no relationship with each other, it can be seen in the missing values heatmap that the missing values does not have any significant correlations.
<br>
Seeing how the missing values only take up a small percentage of the whole dataset and that they have no correlation, checking the distribution before and after dropping the missing values is important. This is because should the distribution has virtually no difference, it is safe to proceed to the next steps after simply dropping the rows with missing values.
<br>
The mean and std of all numerical features does not differ much before and after removing missing values. Thus, it can be assumed that the distribution does not have significant changes.
Looking at the visualistion, there is virtually no difference in the distributions after removing the missing values. With that, the missing values are dropped.

#### Outliers
There were a number of outliers in the numerical dataset. The ones removed are simply the extreme outliers. This is because the model needs to be able to predict unseen data.

### Machine Learning Models
5 models were chosen to create this final predictive model.
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. KNN

### Feature Engineering
1. Split Features and Target
After splitting the features and target, the categorical features are one hot encoded.

2. Multicollinearity
Since one of the models is logistic regression, checking the multicollinearity is important.
The OLS regression results show that there is an indication taht there are strong multicollinearity or other numerical problems.
With that, the next step was to check the VIF.
<br>
The CashbackAmount VIF score is severe at 35. Simultaneously, a feature correlation map was created to see how the relationship between the features.
The feature CashbackAmount has strong business value and is necessary for interpreting customer behaviour, therefore it was kept. However, this decision is noted going forward.

3. Train & Test Split
The dataset is further split into the training and testing sets.

4. Pipelines
The pipelines for both the numerical and categorical features are created.

5. Imbalance
Since the dataset is imbalanced, it handled it using SMOTE in the model pipelines.

### Model Pipelines
All 5 inital models are created pipelines to train and test the data

### Results
Logisitc regression wins recall as it catches 73% of actual churners. Although it has a lower precision score, the ROC AUC of 0.82 shows it has good discriminative power. Since the priority is to minimise churn by catching all potential churners, recall is prioritised over precision. This is also the reason why recall is prioritised over f2 score.

### Hyperparameter Tuning
After hyperparameter tuning, the recall score is still 0.73. This can be due a number of factors. Harder distribution, the number of positive cases is low, the model was close to its best performance, and that logisitc regression is not powerful enough to go beyond that on this data. Therefore, cross validation is needed.

### Cross Validation
The original model CV recall is 0.823 while the tuned model CV recall is 0.826. This means taht the tuned CV recall is higher by only a but and that we reached the limit with logistic regression.
<br>
The second best model is chosen to do hyperparameter tuning to see if it can outperform logistic regression. This model would be the Decision Tree with a recall score of 0.659

#### Hyperparameter Tuning - Decision Tree
The tuned model CV recall is 0.701. Although it performed better than the original Decision Tree model, it still performs worse than the Logistic Regression.
Therefore, the best model in this case is the Logistic Regression.

### Threshold Tuning
Threshold 0.4 seems to be the best case scenario. This is because there is a significant gain in recall as it increases by 0.073 and the f2 score also slightly improves by 0.015. This is at a low cost of precision and f1 score at 0.037 and 0.021 respectively (the values drop but not dramatically)

### Feature Importance
The most important feature is Tenure followed by Prefered Order Category (Laptop & Accessories) then Prefered Order Category (Others).
<br>
Considering that the coefficient for Tenure is -3.78, as tenure increases, the likelihood of churn (class = 1) decreases. In other words, customers with longer tenure are less likely to churn. This makes intuitive sense. A loyal, long-term customer typically have stronger brand engagement.
<br>
Notably, customers that prefers the category laptop and accessories are also less likely to churn while customers who prefer the others category are more likely to churn.

### Deployment & Streamlit
The model is then saved with pickle. It’s important to note that the threshold is not saved and when the model is loaded, the threshold need to be manually set to 0.4.
An app is created through streamlit to predict whether a customer will churn or not churn based on the values inputted by the user for the features.

### Conclusion
#### Model Selection
After testing all 5 models, Logistic Regression is the best-performing classifier for the goal of maximising recall (minimising false negatives), which was critical for the business objective.

#### Performance Metrics
Final model metrics at 0.4 threshold:
- Recall: 0.805
- Precision: 0.333
- F2-Score: 0.627 (optimised for recall)
- ROC AUC: 0.763

This means that the model is highly effective at identifying the postive class (churn = 1), even at the cost of some precision, which aligns with the goal.

#### Threshold Optimisation
A custom threshold of 0.4 was selected over the default 0.5. This improved recall and F2-score, showing better sensitivity toward true positives.

#### Deployment
The model was saved using pickle and a streamlit app was built to enable input of raw data and real-time predictions.

### Recommendation
1. Optimise prediction threshold for business goals
Use threshold 0.4 to maximise true positives and minimise false negatives. This is critical in churn and the company’s business model where each churn can cost a lot (from the customer lifetime value).
2. Use feature importance to inform retention strategy
Tenure has the strongest negative predictor of churn. So it is recommended for the company to target new customers (low tenure) with onboarding programs, loyalty rewards and proactive check ins.
Product preferences like others and mobile are associated with higher churn rates while laptop & accessories buyers churn less. With that, it is recommended to segment and target high-churn product categories with personalised promotions or surveys
It is also recommended to integrate tenure and prefered order category into the CRM for customer lifetime value analysis and segmentation.
3. Evaluate and compare alternate models
Logistic regression performed best overall (a recall score of 0.805 after tuning) but a continued exploration of ensemble models should be considered for further improvement. Especially because it seems like logistic regression has hit it’s limitations.