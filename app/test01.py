import numpy as np
import pandas as pd
import streamlit as st
#streamlit run app/test01.py
#文字顯示
st.title("元件練習操作")
st.header("AAAA")
st.subheader("BBBBB")
st.write("Price:",100)
name="Joe"
st.write("Name: ", name)
st.write("# 元件練習操作")
st.write("## 元件練習操作")
st.write("### 元件練習操作")

a = np.array([10,20,30])
st.write(a)
b = pd.DataFrame([[11,22],[33,44]])
st.write(b)
st.table(b)
st.write(list(range(10)))

# 核取方塊Checkbox
st.write("### 核取方塊Checkbox 1--------------------")
re1 = st.checkbox("白天")
if re1:
    #st.write("day")
    st.info("day")
else:
    #st.write("night")
    st.info("night")

st.write("### 核取方塊Checkbox 2--------------------")
checks = st.columns(4)
with checks[0]:
    c1 = st.checkbox("A")
    if c1:
        st.info("A checked")
with checks[1]:
    c2 = st.checkbox("B")
    if c2:
        st.info("B checked")
with checks[2]:
    c3 = st.checkbox("C")
with checks[3]:
    c4 = st.checkbox("D")

# 選項按鈕RadioButton
st.write("### 選項按鈕RadioButton--------------------------")
sex = st.radio("性別：", ("M","F","None"), index=1)
st.info(sex)

sex2 = st.radio("性別：", ("M","F","None"),key='a')
st.info(sex2)

st.write("### 選項按鈕RadioButton 2+ 數字輸入框") 
col1, col2 = st.columns(2)
with col1:
    num1 = st.number_input("請輸入任一整數：")
with col2:
    num2 = st.number_input("請輸入任一整數：", key='b')
ra = st.radio("計算：", ("+","-","*","/"),key='c')
if ra=='+':
    st.write("{}+{}={}".format(num1,num2,num1+num2))
elif ra=='-':
    st.write("{}-{}={}".format(num1,num2,num1-num2))
elif ra=='*':
    st.write("{}*{}={}".format(num1,num2,num1*num2))
elif ra=='/':
    st.write("{}/{}={}".format(num1,num2,num1/num2))

# 滑桿Slider
st.write("### 滑桿Slider----------------------")
slider = st.slider("請選擇數量：", 1.0, 20.0, step=0.5)
st.info(slider)

slider2 = st.slider("請選擇範圍：", 1.0, 20.0, (11.0,17.0),key="d")
st.info(slider2)

# 下拉選單SelectBox
st.write("### 下拉選單SelectBox:單選")
s1 = st.selectbox("請選擇城市：", ('台北','台中','台南'), index=1)
"選擇的城市：", s1

# 顯示圖片
st.write("### 顯示圖片")
st.image('app/triangle.png')

# 上傳csv
st.write("### 上傳csv")
file = st.file_uploader("請選擇CSV檔")
if file is not None:
    df = pd.read_csv(file, header=None)
    st.write(df.iloc[:10, :])

# 側邊欄SideBar
st.write("### 側邊欄SideBar")
name2 = st.sidebar.text_input("請輸入名稱")
st.sidebar.text(name2)

# 按鈕Button
st.write("### 按鈕Button")
ok = st.button("確定")
if ok:
    st.write("OK")