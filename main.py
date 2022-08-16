import time
import requests

import streamlit as st
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://assets1.lottiefiles.com/packages/lf20_w51pcehl.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_url_dataset = "https://assets7.lottiefiles.com/packages/lf20_a4oisbpo.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)
lottie_dataset = load_lottieurl(lottie_url_dataset)

col1, col2= st.columns(2)

with col1:
    st_lottie(lottie_hello, key="hello")

with col2:
    st.title('Welcome to Machine Learning Web App')


st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer", "Wine")
)

_left,_right = st.columns(2)
with _left:
    st.write(f"## {dataset_name} Dataset")
with _right:
    st_lottie(lottie_dataset, key="dataset",loop=False ,speed=0.5, height=80,width=100)