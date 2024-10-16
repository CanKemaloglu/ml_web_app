import pandas as pd
import requests
import os
import streamlit as st
import numpy as np
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_w51pcehl.json")
lottie_start = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_k8rnjesv.json")

col1, col2 = st.columns(2)
with col1:
    st_lottie(lottie_hello, key="hello", height=200)
with col2:
    st.title('Welcome to Machine Learning Web App')

st.write("## Explore different classifier and datasets\nWhich one is the best?")

# Sidebar Setup
with st.sidebar:
    st_lottie(lottie_start, key="start", height=100, width=300)
    st.sidebar.header("Articles")
    st.sidebar.caption("[PCA using Python (scikit-learn)](https://medium.com/towards-data-science/pca-using-python-scikit-learn-e653f8989e60)")
    st.sidebar.caption("[What is KNN?](https://medium.datadriveninvestor.com/k-nearest-neighbors-knn-algorithm-bd375d14eec7)")
    st.sidebar.caption("[What is SVM?](https://medium.com/towards-data-science/https-medium-com-pupalerushikesh-svm-f4b42800e989)")
    st.sidebar.caption("[What is Random Forest?](https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d)")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

# Dataset and Classifier Info Links
if classifier_name == "KNN":
    st.sidebar.info(f"Go to Scikit Learn [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)")
elif classifier_name == "SVM":
    st.sidebar.info(f"Go to Scikit Learn [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)")
else:
    st.sidebar.info(f"Go to Scikit Learn [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")

# Load Dataset
@st.cache_data
def get_dataset(name):
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    
    X = data.data
    y = data.target
    return
