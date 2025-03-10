# Model Card

For additional information see the [Model Card paper](https://arxiv.org/pdf/1810.03993.pdf).

## Model Details
This model is a RandomForestClassifier that predicts whether an individual’s salary is above or below $50K (binary classification). It uses multiple features such as workclass, education, marital-status, occupation, relationship, race, sex, and native-country.

## Intended Use
The model is intended to aid in understanding the factors that may predict a person’s income threshold. It can be used for educational or demonstration purposes in a machine learning pipeline. Users should be cautious when applying it outside its original context.

## Training Data
The model was trained on a subset of the “census.csv” dataset. The training set was created by randomly splitting the data into 80% training and 20% testing. This dataset contains demographic and occupational attributes as features and a binary salary label as the target.

## Evaluation Data
The test set, comprising 20% of the original dataset, was used for evaluation. The same features and label structure were applied to the test set. Categorical variables were encoded, and continuous variables were used as-is.

## Metrics
The model was evaluated using:
- **Precision**: The proportion of predicted positives that were truly positive.  
- **Recall**: The proportion of actual positives that were identified correctly.  
- **F1 (β=1)**: The harmonic mean of precision and recall.

Typical performance on the test set:  
- Precision: ~0.82  
- Recall: ~0.71  
- F1: ~0.76  

These numbers are examples; please refer to your training logs and metrics file for the exact values.

## Ethical Considerations
1. **Bias and Fairness**: Certain demographic features may lead to unintended biases. Users should conduct fairness checks (e.g., slicing metrics by demographic group) to ensure equitable performance across sub-populations.  
2. **Data Privacy**: The model relies on personal attributes (e.g., native-country, race); appropriate data handling procedures and privacy considerations should be in place when using real-world data.  
3. **Misuse Potential**: Decisions based solely on predicted income level could lead to discrimination in opportunities such as lending, hiring, or housing.

## Caveats and Recommendations
- This model is only as good as the data it was trained on. If the data is incomplete or non-representative, the model’s predictions may not generalize well.  
- Regular re-training is recommended if the population or environment changes.  
- Always combine model outputs with domain expertise and additional validation steps before making high-stakes decisions.