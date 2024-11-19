# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Type: Logistic Regression 
Framework: scikit-learn
Objective: Predict whether an individual earns more than $50K/year based on demographic and employment-related data.
Version: 1.0
Owner: Janae Johnson
Date: 11/19/2024
Input Features: Categorical ( workclass, education, occupation) and numerical features ( age, hours-per-week).

## Intended Use
This model is designed to classify individuals into income brackets:

<=50K: Individuals earning less than or equal to $50K/year.
>50K: Individuals earning more than $50K/year.
Intended Applications:

Understanding income patterns based on demographics.
Assisting in policy-making or resource allocation decisions.
Limitations:

The model may not generalize well to populations outside the original dataset.
Predictions are influenced by biases in the training data.
## Training Data
The model was trained using the Census Income Dataset, which contains data on individuals demographics and employment. Key features include:

Categorical Features: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
Numerical Features: age, hours-per-week, education-num, etc.
The dataset was split into 80% training and 20% testing. Training data consisted of approximately 32,561 and was balanced using stratified sampling based on the income label.
## Evaluation Data
The test data comprised approximately 32,561 and included unseen samples for evaluating the model's performance. Evaluation followed the same preprocessing as the training data.
## Metrics
Overall Metrics:

Precision: 0.7419
Recall: 0.6384
F1-Score: 0.6863
Performance on Slices: The model was evaluated on subsets of data based on unique values of categorical features. For example:

Education = Bachelors:
Precision: 0.75  Recall: 0.65  F1: 0.70
Workclass = Private:
Precision: 0.73  Recall: 0.63  F1: 0.68

## Ethical Considerations
The model’s performance can vary for different groups of people, so it’s important to regularly check how well it works for each group to ensure fairness and avoid reinforcing unfair treatment of any demographic. Since the model uses sensitive information like race and country of origin, it’s crucial to handle the data responsibly and follow privacy standards. Additionally, the model might reflect biases in the data it was trained on, such as historical income inequalities, which could lead to biased predictions for certain groups.
## Caveats and Recommendations
This model should not be the only factor in important decisions like hiring or loan approvals, as it may reflect biases or inaccuracies from the training data. To keep the model fair and accurate, it’s important to regularly check its performance and retrain it when necessary, especially if the data it’s applied to changes over time or differs from the data used to create the model.
