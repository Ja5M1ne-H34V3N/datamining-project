# 完整整合版：PDF 项目全套代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_selector import FeatureSelector
from mlxtend.feature_selection import ColumnSelector
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

# ====================== 1. 读取数据 & 基础检查 ======================
data = pd.read_csv("data_public.csv.gz")
X = data.drop("Class", axis=1)
y = data["Class"]

# 缺失值检查
print("缺失值统计：")
print(data.isnull().sum())

# ====================== 2. 绘制特征直方图（第10页） ======================
for col in X.columns:
    plt.hist(data[col], bins=30)
    plt.title(f"Feature {col}")
    plt.show()

# ====================== 3. 共线性特征筛选（第11页） ======================
fs = FeatureSelector(data=X, labels=y)
fs.identify_collinear(correlation_threshold=0.98)
print("\n共线特征记录：")
print(fs.record_collinear)

# 筛选后保留特征：A,B,D,I,L,M
features_kept = ["A", "B", "D", "I", "L", "M"]

# ====================== 4. LightGBM 特征重要度筛选（第12页） ======================
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X, y)
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\n特征重要度：")
print(importance)

# 最终最优特征：A,B,D
final_features = ["A", "B", "D"]

# ====================== 5. 构建 Pipeline（核心） ======================
pipeline = Pipeline([
    ("selector", ColumnSelector(columns=final_features)),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=1)),
    ("model", RandomForestClassifier(random_state=42))
])

# ====================== 6. 数据集划分 ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ====================== 7. 网格搜索调参 ======================
param_grid = {
    "model__max_depth": [1, 2, 3, 4],
    "model__n_estimators": [1, 5, 10, 20, 50, 100, 1000]
}
grid = GridSearchCV(pipeline, param_grid, cv=4)
grid.fit(X_train, y_train)
print("\n最优参数：", grid.best_params_)

# ====================== 8. 训练最优模型 ======================
best_pipeline = grid.best_estimator_
best_pipeline.fit(X_train, y_train)

# ====================== 9. 模型评估 ======================
y_pred = best_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# ====================== 10. 交叉验证 ======================
cv_scores = cross_val_score(best_pipeline, X, y, cv=6)
print("\n6折交叉验证得分：", cv_scores)
print("交叉验证平均分：", cv_scores.mean())

# ====================== 11. ONNX 转换（尝试） ======================
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as rt

    initial_type = [("float_input", FloatTensorType([None, len(final_features)]))]
    onnx_model = convert_sklearn(best_pipeline, initial_types=initial_type)

    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 验证推理
    sess = rt.InferenceSession("model.onnx")
    input_name = sess.get_inputs()[0].name
    test_np = X_test[final_features].values.astype(np.float32)
    pred_onnx = sess.run(None, {input_name: test_np})[0]
    print("\nONNX 推理成功！")
except Exception as e:
    print("\nONNX 转换失败（原文问题）：", e)