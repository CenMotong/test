import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# 生成房价数据
np.random.seed(42)
X = pd.DataFrame({
    "area": np.random.randint(50, 200, 1000),
    "rooms": np.random.randint(1, 5, 1000)
})
y = X["area"] * 3000 + X["rooms"] * 50000 + np.random.randint(-20000, 20000, 1000)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 评估模型
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"训练集 R² 分数: {train_score:.4f}")
print(f"测试集 R² 分数: {test_score:.4f}")

# 保存模型
joblib.dump(model, "house_price_model.pkl")
print("✅ 模型已保存为 `house_price_model.pkl`")