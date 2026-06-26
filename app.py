import pandas as pd 
import streamlit as st
from sklearn.cluster import KMeans 
from kneed import KneeLocator 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score 
st.set_page_config(page_icon="🛃", page_title="Customer Segmentation", layout="wide")

# load the data
file = st.file_uploader("", type=["csv"])
df=None 
if file:
    df = pd.read_csv(file)
    # st.dataframe(df.head())
with st.sidebar:
    st.title("Customer Segmentation")
    if df is not None:
        features = st.multiselect("Select Features: ", options=df.columns, default=["Annual Income (k$)", "Spending Score (1-100)"])
        df = df.loc[:, features]
def preprocessing(df):
    # label encoding 
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])
    return df 
def elbow():
    # Elbow method to find the optimal number of clusters
    out = []
    k_values = range(1,11)
    for i in k_values:
        model = KMeans(n_clusters=i)
        model.fit(df)
        out.append(model.inertia_)
    KL = KneeLocator(k_values, out, curve="convex", direction="decreasing")
    df1 = pd.DataFrame({"k_values": k_values, "inertia": out})
    st.subheader("Elbow Curve")
    fig = st.line_chart(data=df1, x="k_values", y="inertia")
    return KL.elbow
if df is not None:
    st.subheader("Sample data")
    st.write(df.sample(10))
    df=preprocessing(df)
    # optimized k value
    k = elbow()
    # model training
    model = KMeans(n_clusters=k)
    model.fit(df)
    labels = model.labels_
    df['clusters'] = labels
    # Visualization of clusters 
    st.subheader("Clustered Data")
    st.scatter_chart(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", color="clusters")