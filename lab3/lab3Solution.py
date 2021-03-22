import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        print(self.pima.head())
        self.X_test = None
        self.y_test = None
        

    def define_feature(self):
        feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y

#applying logistic regression on a more relevant set of features
    def define_feature2(self):
        feature_cols = ['glucose', 'skin', 'insulin', 'bmi', 'pedigree']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def train(self):
        # split X and y into training and testing sets
        X, y = self.define_feature()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(solver='lbfgs')
        logreg.fit(X_train, y_train)
        return logreg

#Solution1: training on 70% of the data
    def train1(self):
        # split X and y into training and testing sets
        X, y = self.define_feature()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y,test_size=0.3, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(solver='lbfgs')
        logreg.fit(X_train, y_train)
        return logreg

#Solution2: using more relevant features for training
    def train2(self):
        # split X and y into training and testing sets
        X, y = self.define_feature2()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y,test_size=0.3, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(solver='lbfgs')
        logreg.fit(X_train, y_train)
        return logreg

#Solution3: adding hyperparameters
    def train3(self):
        # split X and y into training and testing sets
        X, y = self.define_feature2()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y,test_size=0.2, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(solver='lbfgs',C=0.7,random_state=42)
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self):
        model = self.train()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def predict1(self):
        model = self.train1()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def predict2(self):
        model = self.train2()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def predict3(self):
        model = self.train3()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    classifer = DiabetesClassifier()
    print(f"Baseline :")
    result = classifer.predict()
    print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")

    print(f"Solution1 :")
    result1 = classifer.predict1()
    score = classifer.calculate_accuracy(result1)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result1)
    print(f"confusion_matrix=${con_matrix}")

    print(f"Solution2 :")
    result2 = classifer.predict2()
    score = classifer.calculate_accuracy(result2)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result2)
    print(f"confusion_matrix=${con_matrix}")

    print(f"Solution3 :")
    result3 = classifer.predict3()
    score = classifer.calculate_accuracy(result3)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result3)
    print(f"confusion_matrix=${con_matrix}")


    
