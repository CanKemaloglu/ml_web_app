import time

import pandas as pd
import requests

import streamlit as st
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import matplotlib.pyplot as plt
from scikit-learn import datasets
from scikit-learn.model_selection import train_test_split
from scikit-learn.decomposition import PCA
from scikit-learn.svm import SVC
from scikit-learn.neighbors import KNeighborsClassifier
from scikit-learn.ensemble import RandomForestClassifier
from scikit-learn.metrics import accuracy_score

@st.cache
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://assets1.lottiefiles.com/packages/lf20_w51pcehl.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_url_dataset = "https://assets7.lottiefiles.com/packages/lf20_a4oisbpo.json"
lottie_url_rocket = "https://assets2.lottiefiles.com/packages/lf20_l3qxn9jy.json"
lottie_url_start = "https://assets7.lottiefiles.com/packages/lf20_k8rnjesv.json"

lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)
lottie_dataset = load_lottieurl(lottie_url_dataset)
lottie_rocket = load_lottieurl(lottie_url_rocket)
lottie_start = load_lottieurl(lottie_url_start)

col1, col2= st.columns(2)
with col1:
    st_lottie(lottie_hello, key="hello")

with col2:
    st.title('Welcome to Machine Learning Web App')

st.write("""
## Explore different classifier and datasets
Which one is the best?
""")

#Articles
with st.sidebar:
    st_lottie(lottie_start, key="start", height=100, width=300, )
st.sidebar.header("Articles")
st.sidebar.caption(f"[PCA using Python (scikit-learn)](https://medium.com/towards-data-science/pca-using-python-scikit-learn-e653f8989e60)")
st.sidebar.caption(f"[What is KNN ?](https://medium.datadriveninvestor.com/k-nearest-neighbors-knn-algorithm-bd375d14eec7)")
st.sidebar.caption(f"[What is SVM ?](https://medium.com/towards-data-science/https-medium-com-pupalerushikesh-svm-f4b42800e989)")
st.sidebar.caption(f"[What is Random Forest ?](https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d)")

# Sidebar Selectboxws

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer", "Wine")
)

classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest")
)

## Sidebar Classifier Info Link



if classifier_name == "KNN":

    classifier_info  = st. sidebar.info(f"Go to Scikit Learn [{classifier_name}](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors) Page")

elif classifier_name == "SVM":
    classifier_info = st.sidebar.info(
        f"Go to Scikit Learn [{classifier_name}](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svm) Page")

else:
    classifier_info = st.sidebar.info(
        f"Go to Scikit Learn [{classifier_name}](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) Page")


# Show the selected ones with check mark

_left, _right = st.columns(2)
with _left:
    st.write(f"### Dataset: {dataset_name} ")
with _right:
    st_lottie(lottie_dataset, key="dataset", loop=False, speed=0.5, height=50, width=100)

_left1, _right2 = st.columns(2)
with _left1:
    st.write(f"### Classifier: {classifier_name} ")
with _right2:
    st_lottie(lottie_dataset, key="classifier", loop=False, speed=0.5, height=50, width=100)

#Getting data ready

def get_dataset(name):
    data = None
    if name == "Iris":
        data = datasets.load_iris()
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        if st.checkbox("Show Dataset"):
            st.write(df)
    elif name == "Wine":
        data = datasets.load_wine()
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        if st.checkbox("Show Dataset"):
            st.write(df)
    else:
        data = datasets.load_breast_cancer()
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        if st.checkbox("Show Dataset"):
            st.write(df)
    X = data.data
    y = data.target
    return X,y

st_lottie(lottie_rocket, key="rocket", speed=0.8, height=150, width=600)

X,y = get_dataset(dataset_name)
shape_col, class_col = st.columns(2)
with shape_col:
    st.write("### Shape of dataset", X.shape, )
with class_col:
    st.write("### Number of classes:", len(np.unique(y)))

#Getting parameters & classifiers ready

st.sidebar.header("Parameters")
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'### Accuracy =', acc)

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.3,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)


























