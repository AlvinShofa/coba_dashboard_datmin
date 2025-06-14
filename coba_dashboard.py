import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, silhouette_score

st.set_page_config(page_title="Disney Princess Dashboard", layout="wide")
st.title("👑 Disney Princess Popularity Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")

df = load_data()

# ---------------- DATA UNDERSTANDING ----------------
st.header("📘 Data Understanding")

with st.expander("🔍 Informasi Umum Dataset"):
    st.write(df.dtypes)
    st.write(f"Jumlah baris: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")
    st.write("Jumlah data null:")
    st.dataframe(df.isnull().sum())

with st.expander("📈 Statistik Deskriptif"):
    st.dataframe(df.describe())

with st.expander("📊 Korelasi Antar Variabel Numerik"):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- BOX PLOT OUTLIERS ----------------
st.header("📦 Deteksi Outlier - Boxplot")

selected_outlier_features = st.multiselect(
    "Pilih fitur untuk boxplot outlier:",
    numeric_cols,
    default=['PopularityScore', 'GoogleSearchIndex2024', 'BoxOfficeMillions']
)

if selected_outlier_features:
    fig, axes = plt.subplots(1, len(selected_outlier_features), figsize=(5*len(selected_outlier_features), 5))
    if len(selected_outlier_features) == 1:
        axes = [axes]
    for i, col in enumerate(selected_outlier_features):
        sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
        axes[i].set_title(f'Outlier: {col}')
    st.pyplot(fig)
else:
    st.info("Pilih minimal satu fitur untuk boxplot.")

# ---------------- K-MEANS CLUSTERING ----------------
st.header("🔍 Unsupervised Learning - K-Means Clustering")

clustering_features = st.multiselect(
    "Pilih fitur untuk clustering:",
    ['PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions'],
    default=['PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions']
)

if clustering_features:
    data_cluster = df[clustering_features].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_cluster)

    st.subheader("📈 Elbow Method")
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        km.fit(data_scaled)
        distortions.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(K_range, distortions, marker='o')
    ax_elbow.set_xlabel("Jumlah Klaster (k)")
    ax_elbow.set_ylabel("Inertia / SSE")
    ax_elbow.set_title("Elbow Method untuk Menentukan k")
    st.pyplot(fig_elbow)

    k = st.slider("Pilih jumlah klaster (k):", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(data_scaled)

    df_clustered = data_cluster.copy()
    df_clustered["Cluster"] = cluster_labels
    silhouette = silhouette_score(data_scaled, cluster_labels)
    st.success(f"Silhouette Score untuk k={k}: **{silhouette:.4f}**")

    st.subheader("📊 Visualisasi Cluster (PCA 2D) + Centroid")

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_scaled)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels

    pca_centroids = pca.transform(kmeans.cluster_centers_)

    cluster_palette = sns.color_palette("Set2", k)
    fig, ax = plt.subplots()
    for cluster_num in range(k):
        cluster_data = pca_df[pca_df['Cluster'] == cluster_num]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'],
                   label=f'Cluster {cluster_num}', s=60,
                   color=cluster_palette[cluster_num])
    # Centroid
    ax.scatter(pca_centroids[:, 0], pca_centroids[:, 1],
               marker='X', color='black', s=200, label='Centroids')

    ax.set_title("Visualisasi Klaster (PCA 2D)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📌 Statistik Tiap Cluster")
    st.dataframe(df_clustered.groupby("Cluster")[clustering_features].mean())

    st.subheader("📁 Data dengan Label Klaster")
    df["Cluster"] = cluster_labels
    st.dataframe(df[['PrincessName'] + clustering_features + ['Cluster']])
else:
    st.warning("Pilih fitur untuk melanjutkan clustering.")

# ---------------- LOGISTIC REGRESSION ----------------
st.header("📉 Supervised Learning - Logistic Regression")

df['IsIconic'] = df['IsIconic'].map({'Yes': 1, 'No': 0})
regression_features = [
    'PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore',
    'BoxOfficeMillions', 'IMDB_Rating', 'AvgScreenTimeMinutes',
    'NumMerchItemsOnAmazon', 'InstagramFanPages', 'TikTokHashtagViewsMillions'
]
X = df[regression_features]
y = df['IsIconic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
y_proba = logreg.predict_proba(X_test_scaled)[:, 1]

st.subheader("🎯 Evaluasi Model")
st.write(f"Akurasi: **{logreg.score(X_test_scaled, y_test):.2f}**")
st.text(classification_report(y_test, y_pred))

fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bukan Ikonik', 'Ikonik'], yticklabels=['Bukan Ikonik', 'Ikonik'], ax=ax2)
ax2.set_xlabel('Prediksi')
ax2.set_ylabel('Aktual')
ax2.set_title('Confusion Matrix')
st.pyplot(fig2)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.2f}')
ax3.plot([0, 1], [0, 1], linestyle='--')
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve")
ax3.legend()
st.pyplot(fig3)

coef_df = pd.DataFrame({
    'Fitur': regression_features,
    'Koefisien': logreg.coef_[0],
    'Odds Ratio': np.exp(logreg.coef_[0])
}).sort_values('Odds Ratio', ascending=False)

st.subheader("📈 Pentingnya Fitur (Odds Ratio)")
st.dataframe(coef_df)

fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.barplot(data=coef_df, y='Fitur', x='Koefisien', palette='viridis', ax=ax4)
ax4.set_title("Kontribusi Fitur dalam Prediksi")
st.pyplot(fig4)
