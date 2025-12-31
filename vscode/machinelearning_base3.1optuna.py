def objective(trial):

    hidden_layer_sizes = trial.suggest_categorical(
        'hidden_layer_sizes',
        [(50,), (100,), (100, 50), (100, 100)]
    )
    alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
    learning_rate_init = trial.suggest_float(
        'learning_rate_init', 1e-4, 1e-2, log=True
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=300,
            random_state=42,
            early_stopping=True
        ))
    ])

    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    aucs = []

    for train_idx, val_idx in sgkf.split(X, y, groups):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_train_cv, y_train_cv)
        y_prob_val = pipe.predict_proba(X_val_cv)[:, 1]

        aucs.append(roc_auc_score(y_val_cv, y_prob_val))

    return np.mean(aucs)
