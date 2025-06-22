import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import os

# ==== 1. Read data ====
features = pd.read_csv('data/features/features.csv').values  # shape [N, D]
meta = pd.read_csv('data/processed/meta.csv')

# ==== 2. PCA Dimensionality Reduction ====
pca20 = PCA(n_components=20, random_state=42).fit_transform(features)
print('PCA shape:', pca20.shape)

# ==== 3. UMAP & t-SNE ====
umap2d = umap.UMAP(n_components=2, random_state=42).fit_transform(pca20)
tsne2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(pca20)

# ==== 4. KMeans Clustering ====
sil_scores, best_k = [], 2
for k in range(2, 10):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(pca20)
    sil = silhouette_score(pca20, labels)
    sil_scores.append(sil)
    if sil == max(sil_scores):
        best_k = k
print(f"KMeans: Best cluster number (k) = {best_k}, silhouette = {max(sil_scores):.3f}")
cluster_labels = KMeans(n_clusters=best_k, random_state=42).fit_predict(pca20)

# ==== 5. DBSCAN Clustering ====
dbscan_labels = DBSCAN(eps=3, min_samples=5).fit_predict(pca20)

# ==== 6. Isolation Forest for Outlier Detection ====
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_flag = iso.fit_predict(pca20) == -1  # True for outliers, False for inliers

# ==== 7. Visualization Function ====
def plot_embed(xy, color, title, color_label, out_file=None):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(color)
    for label in unique_labels:
        idx = (color == label)
        plt.scatter(xy[idx, 0], xy[idx, 1], label=str(label), alpha=0.7, s=35)
    plt.legend(title=color_label, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=200)
    plt.show()

os.makedirs('data/features/plots', exist_ok=True)

# ==== 可视化标签改为 patient_id / slice_idx / cluster / outlier ====
plot_embed(umap2d, meta['patient_id'], "UMAP by Patient ID", "Patient ID", 'data/features/plots/umap_by_patient.png')
plot_embed(tsne2d, meta['slice_idx'], "t-SNE by Slice Index", "Slice Index", 'data/features/plots/tsne_by_sliceidx.png')
plot_embed(tsne2d, cluster_labels, "t-SNE by KMeans Cluster", "KMeans Cluster", 'data/features/plots/tsne_by_kmeans.png')
plot_embed(tsne2d, dbscan_labels, "t-SNE by DBSCAN Cluster", "DBSCAN Cluster", 'data/features/plots/tsne_by_dbscan.png')
plot_embed(tsne2d, outlier_flag, "t-SNE by Outlier Detection", "Outlier", 'data/features/plots/tsne_by_outlier.png')

# ==== 8. KMeans Cluster Feature Mean ====
meta['kmeans_cluster'] = cluster_labels
feat_df = pd.DataFrame(features)
grouped = feat_df.groupby(meta['kmeans_cluster']).mean()
grouped.to_csv('data/features/kmeans_cluster_feature_mean.csv')
print("KMeans cluster feature means have been saved.")

# ==== 9. Important Feature Distribution Statistics and Visualization (e.g., Top 10 Features) ====
important_feat_idx = np.argsort(feat_df.var(axis=0))[-10:]  # Top 10 features with highest variance
for idx in important_feat_idx:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=meta['kmeans_cluster'], y=feat_df[idx])
    plt.title(f'Feature {idx} distribution by KMeans Cluster')
    plt.xlabel('KMeans Cluster')
    plt.ylabel(f'Feature {idx} value')
    plt.tight_layout()
    plt.savefig(f'data/features/plots/boxplot_feature_{idx}.png', dpi=200)
    plt.close()

print("Important feature distribution visualizations have been saved.")

# ==== 10. (Optional) Automatic Clustering Number and Silhouette Score Curve ====
plt.figure(figsize=(6,4))
plt.plot(range(2,10), sil_scores, marker='o')
plt.title('Silhouette Score by Number of Clusters (KMeans)')
plt.xlabel('n_clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig('data/features/plots/kmeans_silhouette_curve.png', dpi=200)
plt.close()
print("KMeans silhouette curve has been saved.")

print("All advanced feature exploration analyses are complete! Resulting images have been saved to data/features/plots/")
