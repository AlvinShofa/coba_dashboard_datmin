import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA

st.set_page_config(page_title="Disney Princess Dashboard", layout="wide")
st.title("👑 Disney Princess Popularity Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")

df = load_data()

# ---------------------------- DASHBOARD UMUM ----------------------------
st.header("📊 Ringkasan Dataset")
st.dataframe(df.head())

st.subheader("🔢 Boxplot untuk Deteksi Outlier")
fig_box, ax_box = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df[['PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions']], ax=ax_box)
ax_box.set_title("Boxplot untuk Mendeteksi Outlier pada Fitur-Fitur Utama")
st.pyplot(fig_box)

# ---------------------------- UNSUPERVISED LEARNING (K-MEANS) ----------------------------
st.header("🔍 Unsupervised Learning - K-Means Clustering")

selected_cols = [
    'PopularityScore',
    'GoogleSearchIndex2024',
    'RottenTomatoesScore',
    'BoxOfficeMillions'
]

data_cluster = df[selected_cols].dropna()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_cluster)
df_scaled = pd.DataFrame(data_scaled, columns=selected_cols)

st.subheader("🔢 Elbow Method untuk Menentukan Jumlah Cluster")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(range(1, 11), wcss, marker='o')
ax_elbow.set_title('Metode Elbow untuk Menentukan Jumlah Cluster')
ax_elbow.set_xlabel('Jumlah Cluster')
ax_elbow.set_ylabel('WCSS')
st.pyplot(fig_elbow)

n_clusters = 3
kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = kmeans_final.fit_predict(data_scaled)
df_result = df.copy()
df_result['Cluster'] = labels

cluster_names = {
    0: "Viral Sensation",
    1: "Classic Icons",
    2: "Underrated Gems"
}
df_result['Cluster Label'] = df_result['Cluster'].map(cluster_names)

st.subheader("📁 Hasil Clustering")
st.dataframe(df_result[['PrincessName'] + selected_cols + ['Cluster', 'Cluster Label']])

st.subheader("📌 Statistik Tiap Cluster")
st.write(df_result.groupby('Cluster Label')[selected_cols].mean())

st.subheader("🗭 Visualisasi Clustering Berdasarkan PCA")
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
centroids_pca = pca.transform(kmeans_final.cluster_centers_)

fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
scatter = ax_pca.scatter(
    data_pca[:, 0], data_pca[:, 1],
    c=labels, cmap='viridis', s=60, edgecolor='k'
)
ax_pca.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1],
    c='red', marker='X', s=200, label='Centroid'
)

ax_pca.set_xlabel("PCA Component 1")
ax_pca.set_ylabel("PCA Component 2")
ax_pca.set_title("Visualisasi Cluster dengan PCA")

unique_labels = sorted(set(labels))
colors = scatter.cmap(scatter.norm(unique_labels))
from matplotlib.colors import to_hex
hex_colors = [to_hex(c) for c in colors]

for i, hex_color in zip(unique_labels, hex_colors):
    ax_pca.scatter([], [], c=hex_color, label=f'Cluster {i} - {cluster_names[i]}')

ax_pca.legend()
ax_pca.grid(True)
st.pyplot(fig_pca)

# ---------------------------- SUPERVISED LEARNING (LOGISTIC REGRESSION) ----------------------------
st.header("📉 Supervised Learning - Logistic Regression")

st.subheader("🔧 Persiapan Data")
df['IsIconic'] = df['IsIconic'].map({'Yes': 1, 'No': 0})
features_for_regression = [
    'PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions',
    'IMDB_Rating', 'AvgScreenTimeMinutes', 'NumMerchItemsOnAmazon', 'InstagramFanPages',
    'TikTokHashtagViewsMillions']
target = 'IsIconic'
X = df[features_for_regression]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"Jumlah data latih: {X_train.shape[0]}")
st.write(f"Jumlah data uji: {X_test.shape[0]}")
st.dataframe(y_train.value_counts().rename("Jumlah").reset_index().rename(columns={"index": "IsIconic"}))
st.dataframe(y_test.value_counts().rename("Jumlah").reset_index().rename(columns={"index": "IsIconic"}))

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.subheader("🔍 Contoh Data Setelah Normalisasi")
st.dataframe(pd.DataFrame(X_train_scaled, columns=features_for_regression).head())

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

st.subheader("🌟 Akurasi Model")
st.write(f"Akurasi pada data uji: {logreg.score(X_test_scaled, y_test):.2f}")

st.subheader("📊 Laporan Klasifikasi")
st.text(classification_report(y_test, y_pred))

st.subheader("📊 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tidak Ikonik', 'Ikonik'],
            yticklabels=['Tidak Ikonik', 'Ikonik'], ax=ax_cm)
ax_cm.set_xlabel('Prediksi')
ax_cm.set_ylabel('Aktual')
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

st.subheader("📊 Kurva ROC dan AUC")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend(loc='lower right')
st.pyplot(fig_roc)

st.subheader("📊 Pentingnya Fitur")
feature_importance = pd.DataFrame({
    'Fitur': features_for_regression,
    'Koefisien': logreg.coef_[0]
})
feature_importance['Rasio_Odds'] = np.exp(feature_importance['Koefisien'])

fig_feat, ax_feat = plt.subplots()
sns.barplot(data=feature_importance, x='Koefisien', y='Fitur', palette='viridis', ax=ax_feat)
ax_feat.set_title('Pentingnya Fitur dalam Memprediksi Status Ikonik')
ax_feat.set_xlabel('Nilai Koefisien')
ax_feat.set_ylabel('Fitur')
st.pyplot(fig_feat)

st.subheader("📈 Rasio Odds (Koefisien Eksponensial)")
st.dataframe(feature_importance.sort_values('Rasio_Odds', ascending=False))
