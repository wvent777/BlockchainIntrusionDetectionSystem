import pandas as pd
import numpy as np
import optunity
import optunity.metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score

# df = datasets.load_digits()
# X = df.data
# y = df.target
# print(X.shape)
# print(y.shape)
# labels = y.tolist()
# print(X)
# print(labels)
X_train = (pd.read_csv('ids_data_py/X_train_hpo.csv')).to_numpy()
y_train = pd.read_csv('ids_data_py/y_train_hpo.csv')
labels = list(y_train.to_numpy().flat)

#Particle Swarm Optimization for Random Forest

# Random Forest Classifier
search_space = {'n_estimators': [10, 100],
                'max_depth': [5, 50],
                'eta': [0.01, 0.9]}

# Define the objective function
@optunity.cross_validated(x=X_train, y=labels, num_folds=5)
def objective(x_train, y_train, x_test, y_test, n_estimators=None, max_depth=None, eta=None):
    # Define the model
    model = xgb.XGBClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth),
                eta=abs(float(eta)), use_label_encoder=False, eval_metric='mlogloss')

    # Fit the model
    # scores = np.mean(cross_val_score(model, x_train, y_train, cv=5, n_jobs=-1,
    #                                  scoring='accuracy'))
    scores = np.mean(cross_val_score(model, x_train, y_train, scoring="accuracy",
                                      cv = StratifiedKFold(shuffle=True, random_state=23333), n_jobs=-1))
    # Return the score
    return scores

# Run the optimization

optimal_config, info, _ = optunity.maximize(objective, solver_name='particle swarm',
                                            num_evals=20, **search_space)
print(optimal_config)
print(info.optimum)
def get_config():
    return optimal_config, info.optimum