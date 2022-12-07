import pandas as pd
import numpy as np
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from FCBF_module import FCBF, FCBFK, FCBFiP, get_i


# Read the original data set sample
# *clean means it comes from the original set
# without metadata removed
print("Reading the original data set sample")
df_clean_sample = pd.read_csv('./ids_data_py/CIC_Collection_clean_sample.csv')
print(df_clean_sample.shape)

# Split Train Set and Test Set
X = df_clean_sample.drop(['ClassLabel'], axis=1).values
y = df_clean_sample.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)
# Feature Selection by Genetic Algorithm 
# clf = SVC(gamma='auto')
# SVC didn't work for large datasets, so I tried SGDClassifier
clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
evolved_estimator = GAFeatureSelectionCV(
    estimator=clf,
    cv=5,
    scoring="accuracy",
    population_size=30,
    generations=15,
    n_jobs=-1,
    verbose=True,
    keep_top_k=10,
    elitism=True,
)

evolved_estimator.fit(X, y)
features = evolved_estimator.best_features_
# First set of best features from genetic algorithm
X_bf = X[:, features]

print(f"Original X Shape: {X.shape}")
print(f"Y Shape: {y.shape}")
print("After Feature Selection by Genetic Algorithm")
print(f"Best Features X Shape: {X_bf.shape}")


# Feature Selection by Fast Correlation Based Filter (FCBF)
fcbf = FCBFK(k = 20)

X_bbf = fcbf.fit_transform(X_bf, y)
print("After Feature Selection by FCBF")
print(X_bbf.shape)
print(y.shape)
print(pd.Series(y).value_counts())

# Oversampling the minority classes
# Adjusting imbalance of original sample data
smote = SMOTE(n_jobs=-1, sampling_strategy='not majority')
print("Oversampling the minority classes of original sample data")
X_bbf, y = smote.fit_resample(X_bbf, y)
print("Class Distribution After Oversampling Minority Classes")
print(pd.Series(y).value_counts())
print(X_bbf.shape)
print(y.shape)
