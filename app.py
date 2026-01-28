import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# --------------------------------------------------
# 1️⃣ App Title & Description
# --------------------------------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)
st.markdown("👉 *Discover hidden customer groups without predefined labels.*")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("Wholesale customers data.csv")

# Drop non-spending columns
df_features = df.drop(columns=["Channel", "Region"])
numeric_columns = df_features.columns.tolist()

# --------------------------------------------------
# 2️⃣ Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1 (Numerical)", numeric_columns, index=0
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2 (Numerical)", numeric_columns, index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)", min_value=2, max_value=10, value=4
)

random_state = st.sidebar.number_input(
    "Random State (Optional)", min_value=0, value=42, step=1
)

run_clustering = st.sidebar.button("🟦 Run Clustering")

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if run_clustering:

    # Ensure two different features are selected
    if feature_1 == feature_2:
        st.error("⚠ Please select two different features.")
    else:
        # --------------------------------------------------
        # Data Preparation
        # --------------------------------------------------
        X = df_features[[feature_1, feature_2]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --------------------------------------------------
        # K-Means Model
        # --------------------------------------------------
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X_scaled)

        df_result = df.copy()
        df_result["Cluster"] = cluster_labels

        # --------------------------------------------------
        # 4️⃣ Visualization Section
        # --------------------------------------------------
        st.subheader("📊 Cluster Visualization")

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=cluster_labels,
            cmap="viridis",
            s=60
        )

        ax.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300,
            c="red",
            marker="X",
            label="Cluster Centers"
        )

        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)
        ax.set_title("Customer Clusters")
        ax.legend()

        st.pyplot(fig)

        # --------------------------------------------------
        # 5️⃣ Cluster Summary Section (FIXED)
        # --------------------------------------------------
        st.subheader("📋 Cluster Summary")

        cluster_counts = (
            df_result
            .groupby("Cluster")
            .size()
            .reset_index(name="Count")
        )

        cluster_means = (
            df_result
            .groupby("Cluster")[[feature_1, feature_2]]
            .mean()
            .reset_index()
        )

        summary = pd.merge(cluster_counts, cluster_means, on="Cluster")

        summary.rename(columns={
            feature_1: f"Avg {feature_1}",
            feature_2: f"Avg {feature_2}"
        }, inplace=True)

        st.dataframe(summary)

        # --------------------------------------------------
        # 6️⃣ Business Interpretation Section
        # --------------------------------------------------
        st.subheader("💡 Business Interpretation")

        avg_f1_mean = summary[f"Avg {feature_1}"].mean()
        avg_f2_mean = summary[f"Avg {feature_2}"].mean()

        for _, row in summary.iterrows():
            cluster_id = int(row["Cluster"])

            if row[f"Avg {feature_1}"] > avg_f1_mean and row[f"Avg {feature_2}"] > avg_f2_mean:
                insight = "High-spending customers across multiple categories"
                emoji = "🟢"
            elif row[f"Avg {feature_1}"] < avg_f1_mean and row[f"Avg {feature_2}"] < avg_f2_mean:
                insight = "Budget-conscious customers with lower spending"
                emoji = "🟡"
            else:
                insight = "Moderate spenders with selective purchasing behavior"
                emoji = "🔵"

            st.markdown(f"**{emoji} Cluster {cluster_id}:** {insight}")

        # --------------------------------------------------
        # 7️⃣ User Guidance Box
        # --------------------------------------------------
        st.info(
            "Customers in the same cluster exhibit similar purchasing behaviour "
            "and can be targeted with similar business strategies."
        )

else:
    st.warning("👈 Select features and click **Run Clustering** to begin.")
