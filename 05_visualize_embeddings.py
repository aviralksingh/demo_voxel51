"""
Script to load saved embeddings and create various visualizations and clusters.

This script:
1. Loads embeddings and sample IDs from disk
2. Applies multiple dimensionality reduction methods (t-SNE, UMAP, PCA)
3. Performs clustering (K-means, DBSCAN)
4. Visualizes the results with matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import umap

# Configuration
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "dinov3_embeddings.npy")
SAMPLE_IDS_FILE = os.path.join(EMBEDDINGS_DIR, "sample_ids.npy")
OUTPUT_DIR = "visualizations"
NUM_CLUSTERS = 10  # Adjust based on your dataset
TSNE_PERPLEXITY = 30  # Adjust for better visualization (typically 5-50)
UMAP_N_NEIGHBORS = 15  # UMAP parameter
UMAP_MIN_DIST = 0.1  # UMAP parameter
RANDOM_STATE = 42
USE_PCA_PREPROCESSING = True  # Use PCA to reduce dimensions before t-SNE/UMAP for speed
PCA_COMPONENTS = 50  # Number of PCA components if preprocessing is enabled

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load embeddings and sample IDs
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}\n"
            f"Please run 04_compute_and_save_embeddings.py first."
        )
    
    embeddings = np.load(EMBEDDINGS_FILE)
    sample_ids = np.load(SAMPLE_IDS_FILE)
    
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Standardize embeddings (optional but recommended for clustering)
    print("Standardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Optional PCA preprocessing for faster computation
    embeddings_for_reduction = embeddings_scaled
    if USE_PCA_PREPROCESSING and embeddings_scaled.shape[1] > PCA_COMPONENTS:
        print(f"Applying PCA preprocessing ({PCA_COMPONENTS} components)...")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        embeddings_for_reduction = pca.fit_transform(embeddings_scaled)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Perform clustering
    print(f"Performing K-means clustering with {NUM_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    print(f"Clustering complete. Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples")
    
    # Perform multiple dimensionality reduction methods
    print("\nComputing dimensionality reductions...")
    
    # 1. t-SNE
    print("  1. Computing t-SNE embedding (this may take a while)...")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=RANDOM_STATE,
        max_iter=1000,
        verbose=0
    )
    embeddings_tsne = tsne.fit_transform(embeddings_for_reduction)
    print("     t-SNE complete!")
    
    # 2. UMAP
    print("  2. Computing UMAP embedding...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
        verbose=True
    )
    embeddings_umap = reducer.fit_transform(embeddings_for_reduction)
    print("     UMAP complete!")
    
    # 3. PCA (2D)
    print("  3. Computing PCA (2D)...")
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    embeddings_pca = pca_2d.fit_transform(embeddings_scaled)
    print(f"     PCA complete! Explained variance: {pca_2d.explained_variance_ratio_.sum():.2%}")
    
    print("All dimensionality reductions complete!")
    
    # Create comprehensive visualization
    print("Creating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    def plot_embedding(ax, embeddings_2d, title, cluster_labels):
        """Helper function to plot embeddings colored by clusters"""
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=cluster_labels,
            cmap='tab20',
            alpha=0.6,
            s=10
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Cluster ID')
    
    # Row 1: Different dimensionality reduction methods
    plot_embedding(axes[0, 0], embeddings_tsne, f't-SNE (perplexity={TSNE_PERPLEXITY})', cluster_labels)
    plot_embedding(axes[0, 1], embeddings_umap, f'UMAP (n_neighbors={UMAP_N_NEIGHBORS})', cluster_labels)
    plot_embedding(axes[0, 2], embeddings_pca, f'PCA (2D)', cluster_labels)
    
    # Row 2: t-SNE with different views
    # t-SNE with cluster centers
    axes[1, 0].scatter(
        embeddings_tsne[:, 0],
        embeddings_tsne[:, 1],
        c=cluster_labels,
        cmap='tab20',
        alpha=0.4,
        s=10
    )
    cluster_centers_2d = []
    for cluster_id in range(NUM_CLUSTERS):
        cluster_mask = cluster_labels == cluster_id
        if np.any(cluster_mask):
            cluster_center_2d = embeddings_tsne[cluster_mask].mean(axis=0)
            cluster_centers_2d.append(cluster_center_2d)
    cluster_centers_2d = np.array(cluster_centers_2d)
    axes[1, 0].scatter(
        cluster_centers_2d[:, 0],
        cluster_centers_2d[:, 1],
        c='red',
        marker='x',
        s=200,
        linewidths=3,
        label='Cluster Centers'
    )
    axes[1, 0].set_title('t-SNE with Cluster Centers', fontsize=12)
    axes[1, 0].set_xlabel('t-SNE dimension 1', fontsize=10)
    axes[1, 0].set_ylabel('t-SNE dimension 2', fontsize=10)
    axes[1, 0].legend()
    
    # UMAP with cluster centers
    axes[1, 1].scatter(
        embeddings_umap[:, 0],
        embeddings_umap[:, 1],
        c=cluster_labels,
        cmap='tab20',
        alpha=0.4,
        s=10
    )
    cluster_centers_umap = []
    for cluster_id in range(NUM_CLUSTERS):
        cluster_mask = cluster_labels == cluster_id
        if np.any(cluster_mask):
            cluster_center_umap = embeddings_umap[cluster_mask].mean(axis=0)
            cluster_centers_umap.append(cluster_center_umap)
    cluster_centers_umap = np.array(cluster_centers_umap)
    axes[1, 1].scatter(
        cluster_centers_umap[:, 0],
        cluster_centers_umap[:, 1],
        c='red',
        marker='x',
        s=200,
        linewidths=3,
        label='Cluster Centers'
    )
    axes[1, 1].set_title('UMAP with Cluster Centers', fontsize=12)
    axes[1, 1].set_xlabel('UMAP dimension 1', fontsize=10)
    axes[1, 1].set_ylabel('UMAP dimension 2', fontsize=10)
    axes[1, 1].legend()
    
    # Cluster size distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    axes[1, 2].bar(unique, counts, color=plt.cm.tab20(unique / max(unique)))
    axes[1, 2].set_title('Cluster Size Distribution', fontsize=12)
    axes[1, 2].set_xlabel('Cluster ID', fontsize=10)
    axes[1, 2].set_ylabel('Number of Samples', fontsize=10)
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualizations
    output_file = os.path.join(OUTPUT_DIR, "all_visualizations.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Save all results
    cluster_assignments_file = os.path.join(OUTPUT_DIR, "cluster_assignments.npy")
    np.save(cluster_assignments_file, cluster_labels)
    print(f"Cluster assignments saved to {cluster_assignments_file}")
    
    tsne_coords_file = os.path.join(OUTPUT_DIR, "tsne_coordinates.npy")
    np.save(tsne_coords_file, embeddings_tsne)
    print(f"t-SNE coordinates saved to {tsne_coords_file}")
    
    umap_coords_file = os.path.join(OUTPUT_DIR, "umap_coordinates.npy")
    np.save(umap_coords_file, embeddings_umap)
    print(f"UMAP coordinates saved to {umap_coords_file}")
    
    pca_coords_file = os.path.join(OUTPUT_DIR, "pca_coordinates.npy")
    np.save(pca_coords_file, embeddings_pca)
    print(f"PCA coordinates saved to {pca_coords_file}")
    
    # Show plot
    plt.show()
    
    print("\nDone! Summary:")
    print(f"  - Total samples: {len(embeddings)}")
    print(f"  - Number of clusters: {NUM_CLUSTERS}")
    print(f"  - Visualization methods: t-SNE, UMAP, PCA")
    print(f"  - Visualization: {output_file}")
    print("\nOther visualization methods available:")
    print("  - t-SNE: Good for local structure, slower")
    print("  - UMAP: Good balance of speed and quality, preserves global structure")
    print("  - PCA: Fast, linear transformation, good for initial exploration")


if __name__ == "__main__":
    main()

