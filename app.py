import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("📦 Wholesale Customer Segmentation")

# Load data
df = pd.read_csv("Wholesale customers data.csv")
st.subheader("Raw Data")
st.dataframe(df.head())

# Feature selection
X = df.drop(columns=['Channel', 'Region'])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose K
k = st.slider("Select number of clusters (K)", 2, 8, 4)

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

st.subheader("Clustered Data")
st.dataframe(df.head())

# Visualization
st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
ax.scatter(X_scaled[:, 2], X_scaled[:, 4], c=clusters, cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 2],
           kmeans.cluster_centers_[:, 4],
           s=300, c='red')
ax.set_xlabel("Grocery")
ax.set_ylabel("Detergents_Paper")
st.pyplot(fig)

# Cluster profiling
st.subheader("Cluster Profiles")
st.dataframe(df.groupby('Cluster').mean())
