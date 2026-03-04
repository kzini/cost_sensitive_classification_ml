import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.model_evaluation import train_and_evaluate_model, make_pipeline

def train_logistic_regression(X_train, X_test, y_train, y_test, model_name, scaling=True):
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    model = make_pipeline(model, scaling=scaling)
    
    return train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        model=model,
        model_name=model_name
    )

def train_random_forest(X_train, X_test, y_train, y_test, model_name):
    model = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    return train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        model=model,
        model_name=model_name
    )

def train_xgboost(X_train, X_test, y_train, y_test, optimize_hyperparams=False, n_iter=100, cv=10):
    # Peso para balancear as classes
    scale_pos_weight = len(y_train) / (2 * np.bincount(y_train)[1])

    
    if optimize_hyperparams:
        xgb = XGBClassifier(objective='binary:logistic', eval_metric='aucpr', random_state=42)
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 2, 3],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'gamma': [0.1, 0.2],
            'reg_lambda': [0.1, 1, 10],
            'reg_alpha': [0.1, 1],
            'scale_pos_weight': [scale_pos_weight, 1]

        }

        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='roc_auc',
            n_jobs=-1,
            cv=cv,
            verbose=3,
            random_state=42,
            refit=True
    )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        model_name = 'XGBoost (Hiperparâmetros otimizados)'

        metrics_df, trained_model = train_and_evaluate_model(
            X_train,
            X_test,
            y_train,
            y_test,
            model=best_model,
            model_name=model_name
        )
        
        return metrics_df, trained_model, search
    
    else:
        best_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42
        )
        model_name = 'XGBoost (Padrão)'

        metrics_df, trained_model = train_and_evaluate_model(
            X_train,
            X_test,
            y_train,
            y_test,
            model=best_model,
            model_name=model_name
        )
        
        return metrics_df, trained_model

def train_svc(X_train, X_test, y_train, y_test, model_name, scaling=True):
    classifier = SVC(kernel='linear', probability=True, random_state=42)
    model = make_pipeline(classifier, scaling=scaling)
    
    return train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        model=model,
        model_name=model_name
    )


def train_knn(X_train, X_test, y_train, y_test, model_name, scaling=True):
    classifier = KNeighborsClassifier(n_neighbors=5)
    model = make_pipeline(classifier, scaling=scaling)

    return train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        model=model,
        model_name=model_name
    )


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def get_cv_roc_auc(model, X, y):
    y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_multiple_roc_auc(model_dict, X, y, title='Curvas ROC - Validação Cruzada'):
    plt.figure(figsize=(8, 6))
    for nome_modelo, modelo in model_dict.items():
        fpr, tpr, roc_auc = get_cv_roc_auc(modelo, X, y)
        plt.plot(fpr, tpr, label=f'{nome_modelo} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=11)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(loc='lower right')
    plt.show()
