import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score
)
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

def evaluate_model_cv(
    X, y, groups,
    base_estimator,
    param_grid,
    need_scaler=False,
    random_state=42
):
    # ===============================
    # CV splitter
    # ===============================
    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    # ===============================
    # GridSearch（パラメータ探索のみ）
    # ===============================
    gs = GridSearchCV(
        base_estimator,
        param_grid,
        cv=sgkf,
        scoring='roc_auc',
        n_jobs=-1
    )
    gs.fit(X, y, groups=groups)
    best_model = gs.best_estimator_

    # ===============================
    # CV 予測を集約
    # ===============================
    all_probs = []
    all_y = []

    roc_aucs = []
    pr_aucs = []

    for tr_idx, va_idx in sgkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        best_model.fit(X_tr, y_tr)
        prob = best_model.predict_proba(X_va)[:, 1]

        all_probs.append(prob)
        all_y.append(y_va.values)

        roc_aucs.append(roc_auc_score(y_va, prob))
        pr_aucs.append(average_precision_score(y_va, prob))

    all_probs = np.concatenate(all_probs)
    all_y = np.concatenate(all_y)

    # ===============================
    # threshold 最適化（F1）
    # ===============================
    thresholds = np.linspace(0, 1, 1001)
    f1s, recalls = [], []

    for th in thresholds:
        pred = (all_probs >= th).astype(int)
        f1s.append(f1_score(all_y, pred))
        recalls.append(recall_score(all_y, pred))

    best_idx = np.argmax(f1s)

    return {
        'cv_roc_auc': np.mean(roc_aucs),
        'cv_pr_auc': np.mean(pr_aucs),
        'cv_f1': f1s[best_idx],
        'cv_recall': recalls[best_idx],
        'threshold': thresholds[best_idx],
        'best_params': gs.best_params_
    }
