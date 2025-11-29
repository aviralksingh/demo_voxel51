"""
Interactive visualization script with hover functionality.

This script:
1. Loads embeddings and sample IDs from disk
2. Loads the FiftyOne dataset to access images
3. Creates an interactive Plotly visualization with:
   - t-SNE/UMAP/PCA embeddings
   - Hover tooltips showing sample ID
   - Image preview on hover
   - Cluster coloring
"""

import os
import base64
import numpy as np
import fiftyone as fo
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuration
DATASET_NAME = "coco-2017-validation"
EMBEDDINGS_DIR = "embeddings"
VISUALIZATIONS_DIR = "visualizations"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "dinov3_embeddings.npy")
SAMPLE_IDS_FILE = os.path.join(EMBEDDINGS_DIR, "sample_ids.npy")
TSNE_COORDS_FILE = os.path.join(VISUALIZATIONS_DIR, "tsne_coordinates.npy")
UMAP_COORDS_FILE = os.path.join(VISUALIZATIONS_DIR, "umap_coordinates.npy")
PCA_COORDS_FILE = os.path.join(VISUALIZATIONS_DIR, "pca_coordinates.npy")
CLUSTER_ASSIGNMENTS_FILE = os.path.join(VISUALIZATIONS_DIR, "cluster_assignments.npy")

# Visualization settings
NUM_CLUSTERS = 10  # Adjust based on your dataset (should match 05_visualize_embeddings.py)
MAX_SAMPLES_FOR_HOVER = 1000  # Limit samples with image hover for performance
IMAGE_SIZE_HOVER = 150  # Size of hover image in pixels
REDUCTION_METHOD = "tsne"  # Options: "tsne", "umap", "pca"


def encode_image_to_base64(image_path, max_size=IMAGE_SIZE_HOVER):
    """Load and encode an image to base64 for hover tooltip."""
    try:
        img = Image.open(image_path)
        # Resize if too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None


def create_sample_id_to_image_map(dataset, sample_ids):
    """Create a mapping from sample IDs to image paths and base64 encoded images."""
    print("Creating sample ID to image mapping...")
    id_to_path = {}
    id_to_base64 = {}
    
    # Create a lookup dictionary for faster access
    sample_dict = {sample.id: sample for sample in dataset}
    
    processed = 0
    for sample_id in sample_ids:
        if sample_id in sample_dict:
            sample = sample_dict[sample_id]
            if sample.filepath:
                id_to_path[sample_id] = sample.filepath
                # Only encode images for a subset to avoid memory issues
                if processed < MAX_SAMPLES_FOR_HOVER:
                    base64_img = encode_image_to_base64(sample.filepath)
                    if base64_img:
                        id_to_base64[sample_id] = base64_img
                processed += 1
    
    print(f"  Mapped {len(id_to_path)} samples")
    print(f"  Encoded {len(id_to_base64)} images for hover preview")
    return id_to_path, id_to_base64


def create_hover_text(sample_id, cluster_id, has_image):
    """Create hover text for a sample."""
    text = f"<b>Sample ID:</b> {sample_id}<br>"
    text += f"<b>Cluster:</b> {cluster_id}<br>"
    if has_image:
        text += "<b>Hover to see image</b>"
    return text


def main():
    # Load embeddings and sample IDs
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}\n"
            f"Please run 04_compute_and_save_embeddings.py first."
        )
    
    sample_ids = np.load(SAMPLE_IDS_FILE)
    print(f"Loaded {len(sample_ids)} sample IDs")
    
    # Load coordinates based on reduction method
    print(f"\nLoading {REDUCTION_METHOD.upper()} coordinates...")
    if REDUCTION_METHOD == "tsne":
        coords_file = TSNE_COORDS_FILE
        title_suffix = "t-SNE"
    elif REDUCTION_METHOD == "umap":
        coords_file = UMAP_COORDS_FILE
        title_suffix = "UMAP"
    elif REDUCTION_METHOD == "pca":
        coords_file = PCA_COORDS_FILE
        title_suffix = "PCA"
    else:
        raise ValueError(f"Unknown reduction method: {REDUCTION_METHOD}")
    
    if not os.path.exists(coords_file):
        print(f"Warning: {coords_file} not found. Please run 05_visualize_embeddings.py first.")
        print("Falling back to computing t-SNE on the fly (this may take a while)...")
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        embeddings = np.load(EMBEDDINGS_FILE)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        tsne = TSNE(n_components=2, random_state=42, max_iter=1000, verbose=1)
        coordinates = tsne.fit_transform(embeddings_scaled)
        title_suffix = "t-SNE (computed on-the-fly)"
    else:
        coordinates = np.load(coords_file)
    
    print(f"Loaded coordinates: {coordinates.shape}")
    
    # Load cluster assignments
    if os.path.exists(CLUSTER_ASSIGNMENTS_FILE):
        cluster_labels = np.load(CLUSTER_ASSIGNMENTS_FILE)
        print(f"Loaded cluster assignments: {len(cluster_labels)} samples")
    else:
        print("Warning: Cluster assignments not found. Using default clustering...")
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        embeddings = np.load(EMBEDDINGS_FILE)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    # Load dataset to get image paths
    print(f"\nLoading dataset: {DATASET_NAME}...")
    try:
        dataset = fo.load_dataset(DATASET_NAME)
    except:
        print(f"Dataset {DATASET_NAME} not found. Cannot load images for hover.")
        dataset = None
    
    # Create image mappings
    id_to_path = {}
    id_to_base64 = {}
    if dataset is not None:
        id_to_path, id_to_base64 = create_sample_id_to_image_map(dataset, sample_ids)
    
    # Prepare hover data
    print("\nPreparing interactive visualization...")
    hover_texts = []
    hover_images = []
    
    for i, sample_id in enumerate(sample_ids):
        has_image = sample_id in id_to_base64
        hover_text = create_hover_text(sample_id, int(cluster_labels[i]), has_image)
        hover_texts.append(hover_text)
        hover_images.append(id_to_base64.get(sample_id, None))
    
    # Create interactive plot
    print("Creating Plotly figure...")
    
    # Create figure with subplots for different views
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f'Interactive {title_suffix} Visualization',),
        specs=[[{"secondary_y": False}]]
    )
    
    # Create scatter plot
    # Group by cluster for better visualization
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        cluster_coords = coordinates[mask]
        cluster_ids = sample_ids[mask]
        cluster_hover_texts = [hover_texts[i] for i in range(len(sample_ids)) if mask[i]]
        cluster_hover_images = [hover_images[i] for i in range(len(sample_ids)) if mask[i]]
        
        # Create custom hover template
        customdata = []
        for i, sample_id in enumerate(cluster_ids):
            idx = np.where(sample_ids == sample_id)[0][0]
            customdata.append({
                'sample_id': sample_id,
                'cluster': int(cluster_id),
                'image': hover_images[idx]
            })
        
        fig.add_trace(
            go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=[f"Sample: {sid}<br>Cluster: {cluster_id}" for sid in cluster_ids],
                hovertemplate='<b>%{text}</b><extra></extra>',
                marker=dict(
                    size=6,
                    opacity=0.7,
                    color=px.colors.qualitative.Set3[int(cluster_id) % len(px.colors.qualitative.Set3)]
                ),
                customdata=customdata
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'Interactive {title_suffix} Embedding Visualization<br><sub>Hover over points to see sample information</sub>',
        xaxis_title=f'{title_suffix} Dimension 1',
        yaxis_title=f'{title_suffix} Dimension 2',
        height=800,
        showlegend=True,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Add custom JavaScript for image hover (if needed, we'll use Plotly's built-in hover)
    # For now, we'll create a separate HTML file with enhanced hover functionality
    
    # Save as HTML
    output_file = os.path.join(VISUALIZATIONS_DIR, f"interactive_{REDUCTION_METHOD}.html")
    print(f"\nSaving interactive visualization to {output_file}...")
    fig.write_html(output_file)
    
    # Also create an enhanced version with image hover using custom HTML
    print("Creating enhanced version with image hover...")
    create_enhanced_html(
        coordinates, sample_ids, cluster_labels, id_to_base64, 
        title_suffix, output_file.replace('.html', '_enhanced.html')
    )
    
    print(f"\nDone! Interactive visualizations saved:")
    print(f"  - Standard: {output_file}")
    print(f"  - Enhanced (with images): {output_file.replace('.html', '_enhanced.html')}")
    print(f"\nOpen the HTML files in your browser to interact with the visualization!")


def create_enhanced_html(coordinates, sample_ids, cluster_labels, id_to_base64, 
                         title_suffix, output_file):
    """Create an enhanced HTML file with image hover functionality."""
    import json
    
    # Prepare data for JavaScript
    data_points = []
    for i in range(len(sample_ids)):
        data_points.append({
            'x': float(coordinates[i, 0]),
            'y': float(coordinates[i, 1]),
            'sample_id': str(sample_ids[i]),
            'cluster': int(cluster_labels[i]),
            'image': id_to_base64.get(sample_ids[i], None)
        })
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive {title_suffix} Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        #hover-info {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 15px;
            max-width: 300px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: none;
            z-index: 1000;
        }}
        #hover-image {{
            max-width: 250px;
            max-height: 250px;
            margin-top: 10px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>Interactive {title_suffix} Embedding Visualization</h1>
    <p>Hover over points to see sample information and images</p>
    <div id="plotly-div"></div>
    <div id="hover-info">
        <h3 id="hover-title">Sample Information</h3>
        <p><strong>Sample ID:</strong> <span id="hover-sample-id"></span></p>
        <p><strong>Cluster:</strong> <span id="hover-cluster"></span></p>
        <img id="hover-image" src="" alt="Sample image" style="display: none;">
    </div>

    <script>
        const dataPoints = {json.dumps(data_points)};
        
        // Group by cluster
        const traces = [];
        const clusters = [...new Set(dataPoints.map(d => d.cluster))].sort((a, b) => a - b);
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
        
        clusters.forEach((cluster, idx) => {{
            const clusterData = dataPoints.filter(d => d.cluster === cluster);
            traces.push({{
                x: clusterData.map(d => d.x),
                y: clusterData.map(d => d.y),
                mode: 'markers',
                type: 'scatter',
                name: `Cluster ${{cluster}}`,
                marker: {{
                    size: 6,
                    opacity: 0.7,
                    color: colors[idx % colors.length]
                }},
                customdata: clusterData.map(d => ({{
                    sample_id: d.sample_id,
                    cluster: d.cluster,
                    image: d.image
                }}))
            }});
        }});
        
        const layout = {{
            title: 'Interactive {title_suffix} Visualization',
            xaxis: {{ title: '{title_suffix} Dimension 1' }},
            yaxis: {{ title: '{title_suffix} Dimension 2' }},
            height: 800,
            hovermode: 'closest',
            template: 'plotly_white'
        }};
        
        const config = {{
            responsive: true,
            displayModeBar: true
        }};
        
        Plotly.newPlot('plotly-div', traces, layout, config);
        
        // Add hover event
        const plotDiv = document.getElementById('plotly-div');
        const hoverInfo = document.getElementById('hover-info');
        
        plotDiv.on('plotly_hover', function(data) {{
            if (data.points.length > 0) {{
                const point = data.points[0];
                const customData = point.customdata;
                
                document.getElementById('hover-sample-id').textContent = customData.sample_id;
                document.getElementById('hover-cluster').textContent = customData.cluster;
                
                const imgElement = document.getElementById('hover-image');
                if (customData.image) {{
                    imgElement.src = customData.image;
                    imgElement.style.display = 'block';
                }} else {{
                    imgElement.style.display = 'none';
                }}
                
                hoverInfo.style.display = 'block';
                hoverInfo.style.left = (data.event.clientX + 20) + 'px';
                hoverInfo.style.top = (data.event.clientY + 20) + 'px';
            }}
        }});
        
        plotDiv.on('plotly_unhover', function(data) {{
            hoverInfo.style.display = 'none';
        }});
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"  Enhanced HTML saved to {output_file}")


if __name__ == "__main__":
    main()

