import pandas as pd
import numpy as np
from collections import Counter
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from FCBF_module import FCBF, FCBFK, FCBFiP, get_i

# Read the original data set sample
# *clean means it comes from the original set without metadata removed
df_clean_sample = pd.read_csv('./ids_data_py/CIC_Collection_clean_sample.csv')

# Split Train Set and Test Set from clean data
X = df_clean_sample.drop(['ClassLabel'], axis=1).values
y = df_clean_sample.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)
# Feature Selection by Genetic Algorithm
# SVC didn't work for large datasets, so I used SGDClassifier
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
print("Starting Genetic Algorithm Feature Selection")
evolved_estimator.fit(X_train, y_train)
features = evolved_estimator.best_features_
X_bf = X_train[:, features]  # Feature Selection by genetic algorithm

print("\n Results of Genetic Algorithm Feature Selection")
print(f"Training Original X Shape: {X_train.shape}")
print(f"Training Original Y Shape: {y_train.shape}")
print("After Feature Selection by Genetic Algorithm")
print(f"Best Features X Shape: {X_bf.shape}, \n")

# Feature Selection by Fast Correlation Based Filter (FCBF)
print("Starting FCBF Feature Selection\n")
fcbf = FCBFK(k=20)
X_bbf = fcbf.fit_transform(X_bf, y_train)
print(f"Results of FCBF Feature Selection: {X_bbf.shape}")


# Oversampling the minority classes
# Adjusting imbalance of original sample data
print("\n Oversampling the Minority Classes of Original Sample Data")
print("Class Distribution before SMOTE: ")
print(pd.Series(y_train).value_counts())
smote = SMOTE(n_jobs=-1, sampling_strategy={6: 500, 7: 500, 5: 1500, 2: 1500, 1: 1500, 4: 2000})
X_bbf, y_train = smote.fit_resample(X_bbf, y_train)
print("\n Class Distribution After Oversampling Minority Classes")
print(pd.Series(y_train).value_counts())
print(Counter(y_train))
assert X_bbf.shape[0] == y_train.shape[0], "X and y must have the same number of rows"

# Save the clean data sample with selected features and oversampled
train_result = pd.concat([pd.DataFrame(X_bbf), pd.DataFrame(y_train)], axis=1)
train_result.to_csv('./ids_data_py/CIC_clean_train_fe.csv', index=False)

test_result = pd.concat([pd.DataFrame(X_test[:, features]), pd.DataFrame(y_test)], axis=1)
test_result.to_csv('./ids_data_py/CIC_clean_test_fe.csv', index=False)

# SMOTE the improved data set
# Read the improved data set sample
print("\n Reading the improved data set sample")
df_improved_sample = pd.read_csv('./ids_data_py/CIC_Collection_improved_sample.csv')
print(f"Improved data shape: {df_improved_sample.shape}")

# Split Train Set and Test Set from improved data
X_im = df_improved_sample.drop(['ClassLabel'], axis=1).values
y_im = df_improved_sample.iloc[:, -1].values.reshape(-1, 1)
y_im = np.ravel(y_im)

X_train_im, X_test_im, y_train_im, y_test_im = train_test_split(X_im, y_im,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=y_im)

x_1, y_1 = X_train_im, y_train_im
print("Oversampling the Minority Classes of Improved Sample Data")
X_train_im, y_train_im = smote.fit_resample(X_train_im, y_train_im)
print("Class Distribution After Oversampling Minority Classes")
print(pd.Series(y_train_im).value_counts())
print(Counter(y_train_im))

assert X_train_im.shape[0] == y_train_im.shape[0], "X and y rows are not the same size"


# Save the improved data sample with oversampled
train_result_im = pd.concat([pd.DataFrame(X_train_im), pd.DataFrame(y_train_im)], axis=1)
train_result_im.to_csv('./ids_data_py/CIC_improved_train_fe.csv', index=False)

test_result_im = pd.concat([pd.DataFrame(X_test_im), pd.DataFrame(y_test_im)], axis=1)
test_result_im.to_csv('./ids_data_py/CIC_improved_test_fe.csv', index=False)

# # Testing over and under sampling methods
# strategy_over = {6: 1000, 7: 1000, 5: 1000, 2: 1000, 1: 1000, 4: 1000}
# strategy_under = {3: 1000, 0: 5000}
# over = SMOTE(sampling_strategy=strategy_over)
# under = RandomUnderSampler(sampling_strategy=strategy_under)
# steps = [('o', over), ('u', under)]
# pipeline = Pipeline(steps=steps)
# X_train_im_test, y_train_im_test = pipeline.fit_resample(x_1, y_1)


def save_plot(x, y):
    counter = Counter(y)
    for label, _ in counter.items():
        row_ix = np.where(y == label)[0]
        plt.scatter(x[row_ix, 0], x[row_ix, 1], label=str(label))
    plt.legend()
    plt.show()
    return


save_plot(X_train_im, y_train_im)
# save_plot(X_train_im_test, y_train_im_test)
