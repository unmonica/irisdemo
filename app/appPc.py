import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("機器學習分類器")
#側邊欄
data = st.sidebar.selectbox("### 請選擇資料集：", ['IRIS', 'WINE', 'CANCER'])
clf = st.sidebar.selectbox("### 請選擇分類器：", ['SVM', 'KNN', 'RandomForest'])

#下載資料並取得 X, y
def loadData(dd):
    myData = None
    if dd == 'IRIS':
        myData = datasets.load_iris()
    elif dd == 'WINE':
        myData = datasets.load_wine()
    else:
        myData = datasets.load_breast_cancer()

    X = myData.data
    y = myData.target
    yName = myData.target_names
    return X,y,yName

X, y, yName = loadData(data)
st.write("### 資料集結構：", X.shape)
st.write("### 資料集分類數量：", len(np.unique(y)))
st.write("### 資料集分類名稱：")
for i in yName:
    st.write("#### ", i)
st.write("### 資料集前5筆資料：")
st.table(X[:5])

#定義模型的參數
def model(m):
    p={}
    if m=='SVM':
        C = st.sidebar.slider("設定參數C：", 0.01, 10.0)
        p['C'] = C
    elif m=='KNN':
        K = st.sidebar.slider("設定參數K：", 1, 10)
        p['K'] = K   
    else:
        N = st.sidebar.slider("設定樹的數量：", 10,500)
        D = st.sidebar.slider("設定樹的分析層數：",1,100)
        p['N'] = N
        p["D"] = D  
    return p

#建立模型
ps = model(clf)
def myModel(clf, p):
    new_clf = None
    if clf=='SVM':
        new_clf = SVC(C = p['C'])
    elif clf=='KNN':
        new_clf = KNeighborsClassifier(n_neighbors = p['K'])   
    else:
        new_clf = RandomForestClassifier(n_estimators=p["N"],max_depth=p['D'])
    return new_clf  

myclf = myModel(clf, ps)

# 分割訓練,測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

# 進行訓練計算+預測
myclf.fit(X_train, y_train)
y_pred = myclf.predict(X_test)

# 進行評分
acc = accuracy_score(y_test, y_pred)
st.write("### 分類準確性：", acc)

# 降維
pca = PCA(2)
newX = pca.fit_transform(X)

# 繪圖
fig = plt.figure()
plt.scatter(newX[:, 0], newX[:, 1], c=y, alpha=0.7)
#plt.show()
st.pyplot(fig)