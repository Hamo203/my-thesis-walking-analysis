def model_predict_proba(X_input):
    return best_model.predict_proba(X_input)
# background（基準分布）
X_background = shap.sample(X, 20, random_state=42)

# SHAP値を計算する対象（可視化用）
X_explain = shap.sample(X, 50, random_state=0)
explainer = shap.TreeExplainer(best_model)

# shap_values: list[class][sample, feature]
shap_values = explainer.shap_values(X_explain)

# positive class（class=1）
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
elif shap_values.ndim == 3:
    shap_values_pos = shap_values[:, :, 1]
else:
    shap_values_pos = shap_values
print("SHAP値の形:", shap_values_pos.shape) # (50, 15) になるはず
print("データの形:", X_explain.shape)       # (50, 15) になるはず

print("SHAP値の形:", shap_values_pos.shape) 
print("データの形:", X_explain.shape)       

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
