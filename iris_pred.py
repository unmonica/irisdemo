import streamlit as st
import joblib

st.title("IRIS 品種預測")

#載入模型
svm = joblib.load("app/svm_model.joblib")
knn = joblib.load("app/knn_model.joblib")
rf = joblib.load("app/rf_model.joblib")

# 選擇模型
clf = st.selectbox("### 請選擇模型：", ("SVM", "KNN", "RandomForest"))
if clf == "SVM":
    model = svm
elif clf == "KNN":
    model = knn
else:
    model = rf

#設定元件
s1 = st.slider("花萼長度：", 3.0, 8.0, 5.8)
s2 = st.slider("花萼寬度：", 2.0, 5.0, 3.5)
s3 = st.slider("花瓣長度：", 1.0, 7.0, 4.5)
s4 = st.slider("花瓣寬度：", 0.1, 2.6, 1.2)

#使用模型進行預測
labels = ['setosa','versicolor','virginica']

if st.button("進行預測"):
    X = [[s1,s2,s3,s4]]
    y = model.predict(X)
    st.write(y[0])
    st.write("### 預測結果：", labels[y[0]])