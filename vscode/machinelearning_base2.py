import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, recall_score, f1_score, classification_report,roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    


# SVMはスケーリングが命です
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, class_weight='balanced', random_state=42))
])

# 探索するパラメータの範囲
param_grid = {
    'svc__C': [0.1, 1, 10, 100],        # 誤分類のペナルティ
    'svc__gamma': ['scale', 0.01, 0.1, 1] # 境界線の複雑さ
}

# 4. StratifiedGroupKFold による交差検証
sgkf = StratifiedGroupKFold(n_splits=5)
grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=sgkf,
    scoring='roc_auc', # AUCを最大化するパラメータを探す
    n_jobs=-1
)

print("最適パラメータを探索中...")
grid_search.fit(X, y, groups=groups)
# 5. 最適なモデルでホールドアウト評価
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")


cv_thresholds = []
cv_aucs, cv_recalls, cv_f1s = [], [], []
print(f"--- SVM Cross-Validation (5-fold) ---")
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    best_model.fit(X_train_cv, y_train_cv)
    y_prob_val = best_model.predict_proba(X_val_cv)[:, 1]
    #しきい値をvalidationで決める
    fpr, tpr, thresholds = roc_curve(y_val_cv, y_prob_val)
    #例：Recall >= 0.8 を満たす最小 threshold
    idx = np.where(tpr >= 0.8)[0][0]
    threshold = thresholds[idx]
    cv_thresholds.append(threshold)

    # その threshold で予測
    y_pred_val = (y_prob_val >= threshold).astype(int) 

    cv_aucs.append(roc_auc_score(y_val_cv, y_prob_val))
    cv_recalls.append(recall_score(y_val_cv, y_pred_val))
    cv_f1s.append(f1_score(y_val_cv, y_pred_val))
    print(f"Fold {fold}: AUC={cv_aucs[-1]:.4f}, Recall={cv_recalls[-1]:.4f}, F1={cv_f1s[-1]:.4f}")

final_threshold = np.mean(cv_thresholds)
# 5. 平均スコアの算出
print("\n=== CV Average Scores ===")
print(f"Mean ROC-AUC: {np.mean(cv_aucs):.4f}")
print(f"Mean Recall : {np.mean(cv_recalls):.4f}")
print(f"Mean F1-Score: {np.mean(cv_f1s):.4f}")

# 6. 外部ホールドアウトデータでの最終評価
# 全訓練データで再学習
best_model.fit(X, y)
h_prob = best_model.predict_proba(X_holdout)[:, 1]
h_pred = (h_prob >= final_threshold).astype(int)

print("\n=== Holdout Performance ===")
print(f"Holdout AUC    : {roc_auc_score(y_holdout, h_prob):.3f}")
print(f"Holdout Recall : {recall_score(y_holdout, h_pred):.3f}")
print(f"Holdout F1     : {f1_score(y_holdout, h_pred):.3f}")

print("\nClassification Report (Holdout):")
print(classification_report(y_holdout, h_pred))