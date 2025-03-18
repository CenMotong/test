import streamlit as st
import pandas as pd
import joblib

# 加载训练好的模型
model = joblib.load("house_price_model.pkl")

# Streamlit 页面
st.title("🏡 Application of house price prediction")
st.write("Enter house information and predict house prices")

# 用户输入
area = st.number_input("🏠 Housing area（㎡）", min_value=30, max_value=500, value=100)
rooms = st.slider("🛏️ Number of rooms", min_value=1, max_value=10, value=3)

# 预测按钮
if st.button("📊 Predicting housing prices"):
    input_data = pd.DataFrame({"area": [area], "rooms": [rooms]})
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Expected housing price: ￥{prediction:,.2f}")