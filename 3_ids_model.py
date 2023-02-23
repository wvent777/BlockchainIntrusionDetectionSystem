import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score
import xgboost as xgb
from xgboost import plot_importance
import random_forest_hpo as rf_hpo
import xgboost_hpo as xgb_hpo

# Dummy Data Test
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=10, n_informative=8, n_redundant=2, n_classes=6, random_state=1)
df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
X_train, X_test, y_train, y_actual = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)
pd.DataFrame(X_train).to_csv('./ids_data_py/X_train_hpo.csv', index=False)
pd.DataFrame(y_train).to_csv('./ids_data_py/y_train_hpo.csv', index=False)

print("TESTING")
class IntrusionDS():

    # Random Forest Classifier
    def apply_rf(self, X_train, X_test, y_train, y_actual, save=False):
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rf_score = rf.score(X_test, y_actual)
        precision, recall, fscore, support = precision_recall_fscore_support \
            (y_actual, y_pred, average='weighted')

        print("Random Forest")
        print(f"Accuracy: {rf_score}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {fscore}\n")
        print(classification_report(y_actual, y_pred))
        cm = confusion_matrix(y_actual, y_pred)
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt=".0f", linewidth=0.5,
                    linecolor="black", cmap="YlGnBu", ax=ax)
        plt.title("Random Forest - Confusion Matrix")
        plt.xlabel("Y - Predicted")
        plt.ylabel("Y - Actual")
        plt.show()
        if save:
            with open('./results/ids-randomforest/ids_rf_results.txt', 'r+') as f:
                print("Random Forest", file=f)
                print(f"Accuracy: {rf_score}", file=f)
                print(f"Precision: {precision}", file=f)
                print(f"Recall: {recall}", file=f)
                print(f"F1 Score: {fscore}\n", file=f)
                print(classification_report(y_actual, y_pred), file=f)
                f.truncate()
            fig.savefig('./results/ids-randomforest/ids_rf_cm.jpeg')

    # XGBoost Classifier
    def apply_xgb(self, X_train, X_test, y_train, y_actual, save=False):
        xgboost = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False)
        xgboost.fit(X_train, y_train)
        y_pred = xgboost.predict(X_test)
        xgb_score = xgboost.score(X_test, y_actual)
        precision, recall, fscore, support = precision_recall_fscore_support \
            (y_actual, y_pred, average='weighted')

        print("XGBoost")
        print(f"Accuracy: {xgb_score}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {fscore}\n")
        print(classification_report(y_actual, y_pred))
        cm = confusion_matrix(y_actual, y_pred)
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt=".0f", linewidth=0.5,
                    linecolor="black", cmap="YlGnBu", ax=ax)
        plt.title("XGBoost - Confusion Matrix")
        plt.xlabel("Y - Predicted")
        plt.ylabel("Y - Actual")
        plt.show()
        if save:
            with open('./results/ids-xgboost/ids_xgb_results.txt', 'r+') as f:
                print("XGBoost", file=f)
                print(f"Accuracy: {xgb_score}", file=f)
                print(f"Precision: {precision}", file=f)
                print(f"Recall: {recall}", file=f)
                print(f"F1 Score: {fscore}\n", file=f)
                print(classification_report(y_actual, y_pred), file=f)
                f.truncate()
            fig.savefig('./results/ids-xgboost/ids_xgb_cm.jpeg')

    # Linear Discriminant Analysis
    def apply_lda(self, X_train, X_test, y_train, y_actual, save=False):
        lda = LinearDiscriminantAnalysis(solver='svd')
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        lda_score = lda.score(X_test, y_actual)
        precision, recall, fscore, support = precision_recall_fscore_support \
            (y_actual, y_pred, average='weighted')

        print("Linear Discriminant Analysis")
        print(f"Accuracy: {lda_score}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {fscore}\n")
        print(classification_report(y_actual, y_pred))
        cm = confusion_matrix(y_actual, y_pred)
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt=".0f", linewidth=0.5,
                    linecolor="black", cmap="YlGnBu", ax=ax)
        plt.title("Linear Discriminant Analysis - Confusion Matrix")
        plt.xlabel("Y - Predicted")
        plt.ylabel("Y - Actual")
        plt.show()
        if save:
            with open('./results/ids-lda/ids_lda_results.txt', 'w+') as f:
                print("Linear Discriminant Analysis", file=f)
                print(f"Accuracy: {lda_score}", file=f)
                print(f"Precision: {precision}", file=f)
                print(f"Recall: {recall}", file=f)
                print(f"F1 Score: {fscore}\n", file=f)
                print(classification_report(y_actual, y_pred), file=f)
                f.truncate()
            fig.savefig('./results/ids-lda/ids_lda_cm.jpeg')

    # Stacked Classifier - Voting Ensemble
    def fit_stacked(self, X_train, X_test, y_train, y_actual, save=False):
        clf1 = RandomForestClassifier(n_estimators=100)
        clf2 = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False)
        clf3 = LinearDiscriminantAnalysis(solver='svd')

        eclf = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('lda', clf3)], voting='hard')
        eclf.fit(X_train, y_train)

    def optimize_classifiers(self, save=False):
        # Random Forest
        rf_config, rf_acc = rf_hpo.get_config()
        print("Random Forest")
        print("Accuracy: {}".format(rf_acc))
        print(rf_config)

        # XGBoost
        xgb_config, xgb_acc = xgb_hpo.get_config()
        print("XGBoost")
        print("Accuracy: {}".format(xgb_acc))
        print(xgb_config)

        # Linear Discriminant Analysis
        lda_config, lda_acc = hpo.get_config()
        print("Linear Discriminant Analysis")
        print("Accuracy: {}".format(lda_acc))

        if save:
            with open('./results/ids_hpo_config.txt', 'w+') as f:
                print("Hyperparameter Optimization Configurations\n", file=f)
                print("Random Forest", file=f)
                print("Accuracy: {}".format(rf_acc), file=f)
                print(rf_config, file=f)
                print('\n', file=f)
                print("XGBoost", file=f)
                print("Accuracy: {}".format(xgb_acc), file=f)
                print(xgb_config, file=f)
                print('\n', file=f)
                print("Linear Discriminant Analysis", file=f)
                print("Accuracy: {}".format(lda_acc), file=f)
                f.truncate()

# X_train, X_test, y_train, y_actual = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)
# X_train.to_csv('./ids_data_py/X_train_hpo.csv', index=False)
# y_train.to_csv('./ids_data_py/y_train_hpo.csv', index=False)

ids = IntrusionDS()

# ids.apply_lda(X_train, X_test, y_train, y_actual)

ids.optimize_classifiers()

