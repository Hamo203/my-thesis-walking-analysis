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
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=sgkf,
    scoring='roc_auc', # AUCを最大化するパラメータを探す
    n_jobs=-1
)

# 最適パラメータ
print("最適パラメータを探索中...")
grid_search.fit(X, y, groups=groups)
best_params = grid_search.best_params_
print("Best params:", best_params)
print(f"Best CV AUC (GridSearch): {grid_search.best_score_:.4f}")

# 最適パラメータで新たにモデルを構築
best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(
        C=best_params['svc__C'],
        probability=True,
        gamma=best_params['svc__gamma'],
        class_weight='balanced',
        random_state=42))
])

all_val_probs = []
all_val_y = []

cv_aucs = []

print("\n=== 5-fold CV (fixed model) ===")
for fold, (train_idx, val_idx) in enumerate(
    sgkf.split(X, y, groups), 1
):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    best_model.fit(X_train, y_train)
    y_prob = best_model.predict_proba(X_val)[:, 1]

    all_val_probs.append(y_prob)
    all_val_y.append(y_val.values)

    auc = roc_auc_score(y_val, y_prob)
    cv_aucs.append(auc)

    print(f"Fold {fold}: AUC = {auc:.4f}")

all_val_probs = np.concatenate(all_val_probs)
all_val_y = np.concatenate(all_val_y)

print(f"\nMean CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

# =========================================================
# CV全体から threshold を1つ決定
# CVで全 validation の確率を集める -> 全部まとめて1回だけ threshold を決める
#   （例：Recall >= 0.8）
# =========================================================
fpr, tpr, thresholds = roc_curve(all_val_y, all_val_probs)

valid_idx = np.where(tpr >= 0.8)[0]
if len(valid_idx) == 0:
    final_threshold = 0.5
else:
    final_threshold = thresholds[valid_idx[0]]

print(f"\nFinal threshold (Recall >= 0.8): {final_threshold:.4f}")

# CV全体での Recall / F1（参考値）
cv_pred = (all_val_probs >= final_threshold).astype(int)
print("\n=== CV performance at selected threshold ===")
print(f"Recall: {recall_score(all_val_y, cv_pred):.4f}")
print(f"F1    : {f1_score(all_val_y, cv_pred):.4f}")

# =========================================================
# 6. 外部ホールドアウト評価
# =========================================================
best_model.fit(X, y)

h_prob = best_model.predict_proba(X_holdout)[:, 1]
h_pred = (h_prob >= final_threshold).astype(int)

print("\n=== Holdout Performance ===")
print(f"AUC    : {roc_auc_score(y_holdout, h_prob):.4f}")
print(f"Recall : {recall_score(y_holdout, h_pred):.4f}")
print(f"F1     : {f1_score(y_holdout, h_pred):.4f}")

print("\nClassification Report (Holdout)")
print(classification_report(y_holdout, h_pred))
