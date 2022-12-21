import pandas as pd
import numpy as np
from collections import Counter
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from FCBF_module import FCBFK
from sklearn.datasets import make_classification


class feauture_engineering(object):
    def __init__(self, df):
        self.df = df
        self.X = df.drop(['ClassLabel'], axis=1).values
        self.y = df.iloc[:, -1].values.reshape(-1, 1)
        self.y = np.ravel(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                train_size=0.8,
                                                                                test_size=0.2,
                                                                                random_state=0,
                                                                                stratify=self.y)
        self.clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
        self.evolved_estimator = GAFeatureSelectionCV(
            estimator=self.clf,
            cv=5,
            scoring="accuracy",
            population_size=30,
            generations=15,
            n_jobs=-1,
            verbose=True,
            keep_top_k=10,
            elitism=True,
        )
        self.fcbf = FCBFK(k=20)

    def genetic_algorithm(self):
        print("Starting Genetic Algorithm Feature Selection")
        self.evolved_estimator.fit(self.X_train, self.y_train)
        features = self.evolved_estimator.best_features_
        X_bf = self.X_train[:, features]  # Feature Selection by genetic algorithm
        print("\n Results of Genetic Algorithm Feature Selection")
        print(f"Training Original X Shape: {self.X_train.shape}")
        print(f"Training Original Y Shape: {self.y_train.shape}")
        print("After Feature Selection by Genetic Algorithm")
        print(f"Best Features X Shape: {X_bf.shape}, \n")
        return X_bf

    def fcbf_algorithm(self, X_bf):
        print("Starting FCBF Feature Selection\n")
        X_bbf = self.fcbf.fit_transform(X_bf, self.y_train)
        print(f"Results of FCBF Feature Selection: {X_bbf.shape}")
        return X_bbf

    def smote(self, X_bbf):
        # Oversampling the minority classes
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_sm, y_sm = smote.fit_resample(X_bbf, self.y_train)
        print(f"Results of SMOTE: {X_sm.shape}")
        return X_sm, y_sm


# Test Case

X, y = make_classification(n_samples=5000, n_features=10, n_informative=8, n_redundant=2, n_classes=6, random_state=1)
df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)