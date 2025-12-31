def model_predict_proba(X_input):
    return best_model.predict_proba(X_input)
# background（基準分布）
X_background = shap.sample(X, 20, random_state=42)

# SHAP値を計算する対象（可視化用）
X_explain = shap.sample(X, 50, random_state=0)

#explainer = shap.LinearExplainer(best_model, X_background)
actual_model = best_model.named_steps['lr'] 

# LinearExplainer に「Pipeline本体」ではなく「LR本体」を渡す
# X_background もスケーリング済みのデータが必要なため、一度Pipelineで変換します
X_bg_transformed = best_model.named_steps['scaler'].transform(X_background)
X_exp_transformed = best_model.named_steps['scaler'].transform(X_explain)

explainer = shap.LinearExplainer(actual_model, X_bg_transformed)

# shap_values: list[class][sample, feature]
#shap_values = explainer.shap_values(X_explain)
shap_values = explainer.shap_values(X_exp_transformed)
# positive class（class=1）
#shap_values_pos = shap_values[1]
#shap_values_pos = shap_values_pos.T
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
elif shap_values.ndim == 3:
    shap_values_pos = shap_values[:, :, 1]
else:
    shap_values_pos = shap_values
print("SHAP値の形:", shap_values_pos.shape) # (50, 15) になるはず
print("データの形:", X_explain.shape)       # (50, 15) になるはず

print(shap_values_pos.shape)
print(X_explain.shape)



shap.summary_plot(
    shap_values_pos,
    X_explain,
    max_display=20
)
shap.summary_plot(
    shap_values_pos,
    X_explain,
    plot_type="bar",
    max_display=20
)
