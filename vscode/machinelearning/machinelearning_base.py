import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, recall_score, f1_score, classification_report


# 3. モデルの定義
svc_model = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    random_state=42
)

# 4. StratifiedGroupKFold による交差検証
sgkf = StratifiedGroupKFold(n_splits=5)
cv_aucs, cv_recalls, cv_f1s = [], [], []

print(f"--- SVM Cross-Validation (5-fold) ---")
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    svc_model.fit(X_train_cv, y_train_cv)
    y_prob = svc_model.predict_proba(X_val_cv)[:, 1]
    y_pred = svc_model.predict(X_val_cv) #固定しきい値 0.5 を自動的に使用

    cv_aucs.append(roc_auc_score(y_val_cv, y_prob))
    cv_recalls.append(recall_score(y_val_cv, y_pred))
    cv_f1s.append(f1_score(y_val_cv, y_pred))
    print(f"Fold {fold}: AUC={cv_aucs[-1]:.4f}, Recall={cv_recalls[-1]:.4f}, F1={cv_f1s[-1]:.4f}")

# 5. 平均スコアの算出
print("\n=== CV Average Scores ===")
print(f"Mean ROC-AUC: {np.mean(cv_aucs):.4f}")
print(f"Mean Recall : {np.mean(cv_recalls):.4f}")
print(f"Mean F1-Score: {np.mean(cv_f1s):.4f}")

# 6. 外部ホールドアウトデータでの最終評価
svc_model.fit(X, y) # 全訓練データで再学習
h_prob = svc_model.predict_proba(X_holdout)[:, 1]
h_pred = svc_model.predict(X_holdout)

print("\n=== Holdout Performance ===")
print(f"Holdout ROC-AUC: {roc_auc_score(y_holdout, h_prob):.4f}")
print(f"Holdout Recall : {recall_score(y_holdout, h_pred):.4f}")
print(f"Holdout F1-Score: {f1_score(y_holdout, h_pred):.4f}")

print("\nClassification Report (Holdout):")
print(classification_report(y_holdout, h_pred))