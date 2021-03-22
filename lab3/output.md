| Experiement | Accuracy | Confusion Matrix | Comment |
|-------------|----------|------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |
| Solution 1   | 0.6883116883116883  | [[138  19] [ 53  21]] |  Training on 70% of the data to improve prediction quality |
| Solution 2   | 0.7965367965367965  | [[143  14] [ 33  41]] |  Using features 'glucose', 'skin thickness', 'insulin', 'bmi', 'diabetes pedigree function' that are more relevant to the target for training |
| Solution 3   | 0.8246753246753247  | [[97 10] [17 30]] |  Adding hyperparameters in Logistic Regression function and further increasing the training data to 80% for improved performance|
