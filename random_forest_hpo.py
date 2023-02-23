import pandas as pd
import numpy as np
import optunity
import optunity.metrics
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


#Particle Swarm Optimization for Random Forest
# Random Forest Classifier
search_space = {'n_estimators': [10, 100],
                'max_features': [1, 64],
                'max_depth': [5, 50],
                'min_samples_split': [2, 11],
                'min_samples_leaf': [1, 11],
                'criterion': [0, 1]}

# Define the objective function
@optunity.cross_validated(x=X_train, y=labels, num_folds=5)
def objective(x_train, y_train, x_test, y_test, n_estimators=None, max_features=None, max_depth=None,
              min_samples_split=None, min_samples_leaf=None, criterion=None):
    # Define the model
    if criterion <0.5:
        cri = 'gini'
    else:
        cri = 'entropy'
    model = RandomForestClassifier(n_estimators=int(n_estimators), max_features=int(max_features),
                                   max_depth=int(max_depth), min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf),
                                   criterion=cri
                                   )
    # Fit the model
    scores = np.mean(cross_val_score(model, x_train, y_train, cv=5, n_jobs=-1,
                                     scoring='accuracy'))
    # Return the score
    return scores

# Run the optimization
optimal_config, info, _ = optunity.maximize(objective, solver_name='particle swarm',
                                            num_evals=20, **search_space
                                            )
def get_config():
    return optimal_config, info.optimum
