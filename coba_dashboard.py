import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Disney Princess Dashboard", layout="wide")
st.title("👑 Disney Princess Popularity Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")

df = load_data()

# ---------------------------- DASHBOARD UMUM ----------------------------
st.header("📊 Ringkasan Dataset")
st.dataframe(df.head())

# ---------------------------- K-MEANS CLUSTERING ----------------------------
st.header("🔍 K-Means Clustering (Unsupervised Learning)")

all_numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
default_features = ['PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions']

selected_features = st.multiselect(
    "📌 Pilih fitur numerik untuk clustering:",
    all_numeric_cols,
    default=default_features
)

if selected_features:
    X = df[selected_features].dropna()

    if X.empty:
        st.warning("⚠️ Tidak ada data yang tersedia setelah menghapus nilai kosong.")
        st.stop()

    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Slider jumlah cluster
    k = st.slider("🔢 Pilih jumlah cluster (K)", min_value=2, max_value=10, value=3)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_clustered = df.loc[X.index].copy()
    df_clustered["Cluster"] = cluster_labels

    # Silhouette Score
    score = silhouette_score(X_scaled, cluster_labels)
    st.success(f"✅ Silhouette Score: {score:.3f} (Semakin mendekati 1 = semakin baik)")

    # Visualisasi PCA
    st.subheader("📈 Visualisasi Clustering dengan PCA")
    if len(selected_features) >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=cluster_labels, cmap='tab10', s=60, edgecolor='k'
        )
        ax.set_title("Visualisasi PCA dari Hasil Clustering")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Pilih minimal 2 fitur untuk visualisasi.")

    # Statistik tiap cluster
    st.subheader("📊 Statistik Tiap Cluster")
    for feature in selected_features:
        st.markdown(f"#### Statistik: `{feature}`")
        st.dataframe(df_clustered.groupby("Cluster")[feature].describe())

    # Tabel akhir
    st.subheader("📋 Data dengan Label Cluster")
    st.dataframe(df_clustered[["PrincessName"] + selected_features + ["Cluster"]].reset_index(drop=True))
else:
    st.warning("⚠️ Silakan pilih minimal satu fitur numerik terlebih dahulu.")

# ---------------------------- SUPERVISED LEARNING ----------------------------
st.header("📉 Supervised Learning - Logistic Regression")

# Mapping target
st.subheader("🔧 Persiapan Data")
df['IsIconic'] = df['IsIconic'].map({'Yes': 1, 'No': 0})
features_for_regression = [
    'PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions',
    'IMDB_Rating', 'AvgScreenTimeMinutes', 'NumMerchItemsOnAmazon', 'InstagramFanPages',
    'TikTokHashtagViewsMillions']
target = 'IsIconic'

X = df[features_for_regression].dropna()
y = df.loc[X.index, target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Jumlah data latih: {X_train.shape[0]}")
st.write(f"Jumlah data uji: {X_test.shape[0]}")
st.dataframe(y_train.value_counts().rename("Jumlah").reset_index().rename(columns={"index": "IsIconic"}))

# Normalisasi
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Contoh hasil normalisasi
st.subheader("🔍 Contoh Data Setelah Normalisasi")
st.dataframe(pd.DataFrame(X_train_scaled, columns=features_for_regression).head())

# Model Logistic Regression
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# Akurasi
st.subheader("🎯 Akurasi Model")
accuracy = logreg.score(X_test_scaled, y_test)
st.write(f"Akurasi pada data uji: {accuracy:.2f}")

# Classification Report
st.subheader("📋 Laporan Klasifikasi")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("🧮 Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tidak Ikonik', 'Ikonik'],
            yticklabels=['Tidak Ikonik', 'Ikonik'], ax=ax3)
ax3.set_xlabel('Prediksi')
ax3.set_ylabel('Aktual')
ax3.set_title('Confusion Matrix')
st.pyplot(fig3)

# ROC Curve
st.subheader("📊 Kurva ROC dan AUC")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
fig4, ax4 = plt.subplots()
ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
ax4.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax4.set_title("ROC Curve")
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.legend(loc="lower right")
st.pyplot(fig4)

# Feature Importance
st.subheader("📌 Pentingnya Fitur")
feature_importance = pd.DataFrame({
    'Fitur': features_for_regression,
    'Koefisien': logreg.coef_[0]
}).sort_values(by='Koefisien', ascending=False)

fig5, ax5 = plt.subplots()
sns.barplot(data=feature_importance, x='Koefisien', y='Fitur', palette='viridis', ax=ax5)
ax5.set_title("Pentingnya Fitur dalam Memprediksi Status Ikonik")
st.pyplot(fig5)

# Odds Ratio
feature_importance['Rasio_Odds'] = np.exp(feature_importance['Koefisien'])
st.subheader("📈 Rasio Odds (Koefisien Eksponensial)")
st.dataframe(feature_importance.sort_values('Rasio_Odds', ascending=False))
