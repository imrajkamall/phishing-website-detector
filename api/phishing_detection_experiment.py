"""
Phishing Website Detection Using Machine Learning
Full experimental pipeline: baseline classifiers, tuning, ensembling,
and threshold-calibrated XGBoost (proposed technique).

Dataset: UCI Phishing Websites Dataset (Mohammad et al.), 2456 instances, 30 features.
"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, precision_recall_curve)
import xgboost as xgb
import lightgbm as lgb

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load and preprocess data
# ---------------------------------------------------------------
df = pd.read_csv('phishing_dataset.csv')
X = df.drop(columns=['Result']).values
y = np.where(df['Result'].values == 1, 1, 0)  # 1 = phishing, 0 = legitimate

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def evaluate(name, model, X_tr, y_tr, X_te, y_te, proba=None):
    pred = model.predict(X_te)
    if proba is None:
        proba = model.predict_proba(X_te)[:, 1]
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_te, pred),
        'Precision': precision_score(y_te, pred),
        'Recall': recall_score(y_te, pred),
        'F1_Score': f1_score(y_te, pred),
        'ROC_AUC': roc_auc_score(y_te, proba),
    }

results = []

# ---------------------------------------------------------------
# 2. Baseline classifiers
# ---------------------------------------------------------------
baselines = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'SVM': SVC(probability=True, C=2, random_state=RANDOM_STATE),
}
fitted = {}
for name, model in baselines.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    results.append(evaluate(name, model, X_train, y_train, X_test, y_test))

# ---------------------------------------------------------------
# 3. Hyperparameter-tuned Random Forest
# ---------------------------------------------------------------
rf_grid = {'n_estimators': [200, 400], 'max_depth': [None, 20, 30], 'min_samples_split': [2, 4]}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), rf_grid, cv=cv, scoring='f1', n_jobs=-1)
rf_gs.fit(X_train, y_train)
best_rf = rf_gs.best_estimator_
fitted['Random Forest (tuned)'] = best_rf
results.append(evaluate('Random Forest (tuned)', best_rf, X_train, y_train, X_test, y_test))

# ---------------------------------------------------------------
# 4. Gradient Boosting
# ---------------------------------------------------------------
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE)
gb.fit(X_train, y_train)
fitted['Gradient Boosting'] = gb
results.append(evaluate('Gradient Boosting', gb, X_train, y_train, X_test, y_test))

# ---------------------------------------------------------------
# 5. Hyperparameter-tuned XGBoost
# ---------------------------------------------------------------
xgb_grid = {'n_estimators': [200, 400], 'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0]}
xgb_gs = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
                       xgb_grid, cv=cv, scoring='f1', n_jobs=-1)
xgb_gs.fit(X_train, y_train)
best_xgb = xgb_gs.best_estimator_
fitted['XGBoost (tuned)'] = best_xgb
results.append(evaluate('XGBoost (default threshold)', best_xgb, X_train, y_train, X_test, y_test))

# ---------------------------------------------------------------
# 6. LightGBM (tuned)
# ---------------------------------------------------------------
lgb_grid = {'n_estimators': [200, 400], 'max_depth': [-1, 8, 12],
            'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]}
lgb_gs = GridSearchCV(lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                       lgb_grid, cv=cv, scoring='f1', n_jobs=-1)
lgb_gs.fit(X_train, y_train)
best_lgb = lgb_gs.best_estimator_
results.append(evaluate('LightGBM (tuned)', best_lgb, X_train, y_train, X_test, y_test))

# ---------------------------------------------------------------
# 7. Voting / stacking ensembles (alternative core-module designs)
# ---------------------------------------------------------------
ensemble_combos = {
    'Ensemble Voting (RF+GB+XGB)': [('rf', best_rf), ('gb', gb), ('xgb', best_xgb)],
    'Ensemble Voting (RF+GB+KNN)': [('rf', best_rf), ('gb', gb), ('knn', fitted['KNN'])],
}
for name, ests in ensemble_combos.items():
    ens = VotingClassifier(estimators=ests, voting='soft')
    ens.fit(X_train, y_train)
    results.append(evaluate(name, ens, X_train, y_train, X_test, y_test))

stack = StackingClassifier(
    estimators=[('rf', best_rf), ('gb', gb), ('xgb', best_xgb), ('knn', fitted['KNN'])],
    final_estimator=LogisticRegression(max_iter=1000), cv=5
)
stack.fit(X_train, y_train)
results.append(evaluate('Stacking (RF+GB+XGB+KNN)', stack, X_train, y_train, X_test, y_test))

# ---------------------------------------------------------------
# 8. PROPOSED: Threshold-calibrated XGBoost
# ---------------------------------------------------------------
oof_proba = cross_val_predict(best_xgb, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
prec, rec, thresh = precision_recall_curve(y_train, oof_proba)
f1_curve = 2 * prec * rec / (prec + rec + 1e-9)
best_thresh = thresh[np.nanargmax(f1_curve)]

proba_test = best_xgb.predict_proba(X_test)[:, 1]
pred_calibrated = (proba_test >= best_thresh).astype(int)
results.append({
    'Model': 'Threshold-Calibrated XGBoost (Proposed)',
    'Accuracy': accuracy_score(y_test, pred_calibrated),
    'Precision': precision_score(y_test, pred_calibrated),
    'Recall': recall_score(y_test, pred_calibrated),
    'F1_Score': f1_score(y_test, pred_calibrated),
    'ROC_AUC': roc_auc_score(y_test, proba_test),
})

# ---------------------------------------------------------------
# 9. Save results
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values('F1_Score', ascending=False)
results_df.to_csv('experiment_results.csv', index=False)
print(f"\nCalibrated decision threshold (from CV on training set): {best_thresh:.3f}")
print(results_df.to_string(index=False))
