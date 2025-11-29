"""
Semantic search script using DINOv3 embeddings.

This script:
1. Takes an input image path
2. Computes DINOv3 embedding for the input image
3. Searches for similar images in the stored embeddings
4. Fetches the similar images from the dataset
5. Saves the results to a folder in visualizations
"""

import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import fiftyone as fo
import fiftyone.zoo as foz
import transformers
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATASET_NAME = "coco-2017-validation"
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "dinov3_embeddings.npy")
SAMPLE_IDS_FILE = os.path.join(EMBEDDINGS_DIR, "sample_ids.npy")
FILEPATHS_FILE = os.path.join(EMBEDDINGS_DIR, "filepaths.npy")  # Stable identifier
VISUALIZATIONS_DIR = "visualizations"
SEARCH_RESULTS_DIR = os.path.join(VISUALIZATIONS_DIR, "semantic_search_results")
DEFAULT_TOP_K = 10  # Number of similar images to retrieve
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def load_dinov3_model():
    """Load the DINOv3 model and processor."""
    print("Loading DINOv3 model...")
    model = transformers.AutoModel.from_pretrained(MODEL_NAME)
    processor = transformers.AutoImageProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("  Using GPU")
    else:
        print("  Using CPU")
    return model, processor


def compute_embedding_for_image(image_path, model, processor):
    """Compute DINOv3 embedding for a single image."""
    print(f"Computing embedding for: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Process image using the processor
    inputs = processor(images=image, return_tensors="pt")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Compute embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # DINOv3 returns CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embedding[0]  # Return as 1D array


def find_similar_embeddings(query_embedding, stored_embeddings, filepaths, top_k=DEFAULT_TOP_K):
    """Find top-k most similar embeddings using cosine similarity."""
    print(f"Searching for top-{top_k} similar images...")
    
    # Normalize embeddings for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    stored_norm = stored_embeddings / np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarities
    similarities = np.dot(stored_norm, query_norm)
    
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    top_k_similarities = similarities[top_k_indices]
    top_k_filepaths = filepaths[top_k_indices]
    
    print(f"Found {len(top_k_indices)} similar images")
    print(f"Similarity scores range: {top_k_similarities.min():.4f} to {top_k_similarities.max():.4f}")
    
    return top_k_indices, top_k_similarities, top_k_filepaths


def fetch_and_save_images(filepaths, similarities, output_dir, query_image_path=None):
    """Fetch images using filepaths and save them to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nFetching and saving {len(filepaths)} images...")
    
    # Save query image if provided
    if query_image_path and os.path.exists(query_image_path):
        query_output = os.path.join(output_dir, "00_query_image.jpg")
        query_img = Image.open(query_image_path)
        query_img.save(query_output)
        print(f"  Saved query image: {query_output}")
    
    # Save similar images using filepaths (stable identifier)
    saved_count = 0
    for idx, (filepath, similarity) in enumerate(zip(filepaths, similarities), start=1):
        if filepath and os.path.exists(filepath):
            # Extract filename from path for display
            filename_base = os.path.basename(filepath)
            # Create output filename with similarity score
            filename = f"{idx:02d}_similarity_{similarity:.4f}_{filename_base}"
            output_path = os.path.join(output_dir, filename)
            
            # Copy image
            img = Image.open(filepath)
            img.save(output_path)
            saved_count += 1
            print(f"  [{idx}/{len(filepaths)}] Saved: {filename}")
        else:
            print(f"  [{idx}/{len(filepaths)}] Warning: Image not found at {filepath}")
    
    print(f"\nSaved {saved_count} images to {output_dir}")
    return saved_count


def create_summary_file(output_dir, query_image_path, filepaths, similarities, top_k_indices):
    """Create a summary text file with search results."""
    summary_path = os.path.join(output_dir, "search_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Semantic Search Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Query Image: {query_image_path}\n")
        f.write(f"Number of Results: {len(filepaths)}\n\n")
        f.write("Results (sorted by similarity, highest first):\n")
        f.write("-" * 50 + "\n")
        for idx, (filepath, similarity, orig_idx) in enumerate(
            zip(filepaths, similarities, top_k_indices), start=1
        ):
            filename_base = os.path.basename(filepath) if filepath else "N/A"
            f.write(f"{idx}. Filepath: {filepath}\n")
            f.write(f"   Filename: {filename_base}\n")
            f.write(f"   Similarity: {similarity:.6f}\n")
            f.write(f"   Original Index: {orig_idx}\n")
            f.write(f"   Saved as: {idx:02d}_similarity_{similarity:.4f}_{filename_base}\n\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search using DINOv3 embeddings"
    )
    parser.add_argument(
        "image_path",
        default="/home/aviral/fiftyone/voxel51/coco-2017/validation/data/000000000285.jpg",
        type=str,
        help="Path to the input image for semantic search"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of similar images to retrieve (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: visualizations/semantic_search_results/query_name)"
    )
    
    args = parser.parse_args()
    
    # Validate input image
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")
    
    # Create output directory
    if args.output_dir is None:
        query_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_dir = os.path.join(SEARCH_RESULTS_DIR, query_name)
    else:
        output_dir = args.output_dir
    
    print("=" * 60)
    print("Semantic Search using DINOv3 Embeddings")
    print("=" * 60)
    print(f"Query Image: {args.image_path}")
    print(f"Top-K: {args.top_k}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Load stored embeddings
    print("Loading stored embeddings...")
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}\n"
            f"Please run 04_compute_and_save_embeddings.py first."
        )
    
    stored_embeddings = np.load(EMBEDDINGS_FILE)
    
    # Load filepaths (stable identifier) - fallback to sample_ids if filepaths don't exist
    if os.path.exists(FILEPATHS_FILE):
        filepaths = np.load(FILEPATHS_FILE, allow_pickle=True)
        print(f"Loaded {len(filepaths)} filepaths (stable identifiers)")
    else:
        print(f"Warning: {FILEPATHS_FILE} not found.")
        print("Please re-run 04_compute_and_save_embeddings.py to generate filepaths.")
        print("Falling back to sample IDs (may not work if dataset was reloaded)...")
        sample_ids = np.load(SAMPLE_IDS_FILE)
        # Try to get filepaths from dataset if possible
        try:
            dataset_temp = fo.load_dataset(DATASET_NAME)
            sample_dict = {sample.id: sample.filepath for sample in dataset_temp}
            filepaths = np.array([sample_dict.get(sid, None) for sid in sample_ids], dtype=object)
            print(f"  Extracted filepaths from dataset")
        except:
            raise FileNotFoundError(
                f"Filepaths file not found and cannot extract from dataset.\n"
                f"Please re-run 04_compute_and_save_embeddings.py to generate {FILEPATHS_FILE}"
            )
    
    print(f"Loaded {len(stored_embeddings)} embeddings")
    print(f"Embedding dimension: {stored_embeddings.shape[1]}")
    
    # Load DINOv3 model
    model, processor = load_dinov3_model()
    
    # Compute embedding for query image
    query_embedding = compute_embedding_for_image(args.image_path, model, processor)
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Find similar embeddings
    top_k_indices, top_k_similarities, top_k_filepaths = find_similar_embeddings(
        query_embedding, stored_embeddings, filepaths, top_k=args.top_k
    )
    
    # Fetch and save images using filepaths (no need to load dataset)
    saved_count = fetch_and_save_images(
        top_k_filepaths, top_k_similarities, output_dir, args.image_path
    )
    
    # Create summary file
    create_summary_file(
        output_dir, args.image_path, top_k_filepaths, 
        top_k_similarities, top_k_indices
    )
    
    print("\n" + "=" * 60)
    print("Search Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"Total images saved: {saved_count + 1} (including query)")
    print(f"Top similarity score: {top_k_similarities[0]:.6f}")
    print(f"Bottom similarity score: {top_k_similarities[-1]:.6f}")


if __name__ == "__main__":
    main()

