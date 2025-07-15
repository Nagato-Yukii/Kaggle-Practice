import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier
from lightgbm import plot_importance as lgb_plot_importance
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import gc


def objective_xgb(trial, X, y):
    "为xgboost定义超参数搜索空间"
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }

    model = xgb.XGBClassifier(**param)
    return cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

def objective_lgb(trial, X, y):
    "为LightGBM定义超参数搜索空间"
    param = {
        'objective': 'binary',
        'metric': 'accuracy',
        'random_state': 42,
        'verbose': -1,
        
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 50, 250),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    model = LGBMClassifier(**param)
    return cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=1).mean()

def objective_cat(trial, X, y):
    "为CatBoost 定义超参数搜索空间"
    param = {
        'objective': 'Logloss',
        'random_seed': 42,
        'verbose': 0,
        
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 6, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True)
    }
    model = CatBoostClassifier(**param)
    return cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=1).mean()

def train_and_evaluate_xgb(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=30)
    
    xgb_best_params = study.best_params
    print("\nBest Optuna Parameters in xgboost:", xgb_best_params)

    xgb_best_params.update({'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42})

    best_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    best_model.fit(X_train, y_train)
    val_preds = best_model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"\nTuned XGBoost Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, val_preds))

    plt.figure(figsize=(12, 6))
    plot_importance(best_model, max_num_features=15)
    plt.title("Top Feature Importances(XGBoost)")
    plt.tight_layout()
    plt.show()

    return best_model, xgb_best_params

def train_and_evaluate_lgb(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_lgb(trial, X_train, y_train), n_trials=30)

    lgb_best_params = study.best_params
    print("\nBest Optuna Parameters in LightGBM:", lgb_best_params)
    lgb_best_params.update({'objective': 'binary', 'metric': 'accuracy', 'random_state': 42, 'verbose': -1})

    best_model = LGBMClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    best_model.fit(X_train, y_train)
    val_preds = best_model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"\nTuned LightGBM Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, val_preds))

    plt.figure(figsize=(12, 6))
    lgb_plot_importance(best_model, max_num_features=15)
    plt.title("Top Feature Importances(LGBM)")
    plt.tight_layout()
    plt.show()

    return best_model, lgb_best_params

def train_and_evaluate_cat(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_cat(trial, X_train, y_train), n_trials=30)

    cat_best_params = study.best_params
    print("\nBest Optuna Parameters in CatBoost:", cat_best_params)
    cat_best_params.update({'objective': 'Logloss', 'random_seed': 42, 'verbose': 0})

    best_model = CatBoostClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    val_preds = best_model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"\nTuned CaBoost Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, val_preds))

    plt.figure(figsize=(12, 6))

    feat_importances = pd.Series(best_model.get_feature_importance(), index=X.columns)
    feat_importances.nlargest(15).plot(kind='barh')

    plt.title("Top Feature Importances (CatBoost)")
    plt.tight_layout()
    plt.show()

    return best_model, cat_best_params

def stacking(X, y_encoded, X_test, n_splits=10):
    """
    执行Stacking集成学习。
    
    Args:
        xgb_params (dict): XGBoost的最佳超参数。
        lgb_params (dict): LightGBM的最佳超参数。
        cat_params (dict): CatBoost的最佳超参数。
        X (pd.DataFrame): 完整的训练集特征。(feature 4 training)
        y_encoded (np.array): 数值化后的完整训练集标签。(target 4 training)
        X_test (pd.DataFrame): 完整的测试集特征。(features 4 final generate)就是对应submission的
        n_splits (int): 交叉验证的折数。
        
    Returns:
        tuple: (最终的测试集预测概率, OOF预测结果DataFrame)
    """
    cat_model, cat_best_params = train_and_evaluate_cat(X, y_encoded)
    xgb_model, xgb_best_params = train_and_evaluate_xgb(X, y_encoded)
    lgb_model, lgb_best_params = train_and_evaluate_lgb(X, y_encoded)

    models = {

    'cat': CatBoostClassifier(**cat_best_params),
    'xgb': XGBClassifier(**xgb_best_params),
    'lgb': LGBMClassifier(**lgb_best_params)

    }

    oof_preds = {name: np.zeros(len(X)) for name in models.keys()}
    test_preds = {name: np.zeros(len(X_test)) for name in models.keys()}

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
        print(f"--- FOLD {fold+1}/{n_splits} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            oof_preds[name][val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds[name] += model.predict_proba(X_test)[:, 1] / n_splits
        gc.collect()
    
    oof_df = pd.DataFrame(oof_preds)
    test_preds_df = pd.DataFrame(test_preds)
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(oof_df, y_encoded)
    final_predictions = meta_model.predict(test_preds_df)

    return final_predictions
