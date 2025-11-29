"""
Script to compute DINOv3 embeddings for a dataset and save them to disk.

This script:
1. Loads a FiftyOne dataset
2. Computes embeddings using DINOv3 model
3. Saves embeddings and corresponding sample IDs to disk as numpy arrays
"""

import os
import numpy as np
import fiftyone as fo
import fiftyone.utils.transformers as fouhft
import transformers
import fiftyone.zoo as foz

# Configuration
DATASET_NAME = "coco-2017-validation"
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "dinov3_embeddings.npy")
SAMPLE_IDS_FILE = os.path.join(EMBEDDINGS_DIR, "sample_ids.npy")
EMBEDDINGS_FIELD = "embeddings_dinov3"

def main():
    # Create embeddings directory if it doesn't exist
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}")
    try:
        dataset = fo.load_dataset(DATASET_NAME)
    except:
        print(f"Dataset {DATASET_NAME} not found. Loading from zoo...")
        dataset = foz.load_zoo_dataset(
            "https://github.com/voxel51/coco-2017",
            split="validation",
            dataset_name=DATASET_NAME,
        )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Check if embeddings already exist in the dataset
    if EMBEDDINGS_FIELD in dataset.get_field_schema():
        print(f"Embeddings field '{EMBEDDINGS_FIELD}' already exists in dataset.")
        print("Extracting existing embeddings...")
    else:
        # Load DINOv3 model
        print("Loading DINOv3 model...")
        transformers_model = transformers.AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
        model_config = fouhft.FiftyOneTransformerConfig({
            "model": transformers_model,
            "name_or_path": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        })
        model = fouhft.FiftyOneTransformer(model_config)
        
        # Compute embeddings
        print("Computing embeddings (this may take a while)...")
        dataset.compute_embeddings(model, embeddings_field=EMBEDDINGS_FIELD)
        print("Embeddings computed!")
    
    # Extract embeddings and sample IDs
    print("Extracting embeddings from dataset...")
    embeddings_list = []
    sample_ids_list = []
    
    for sample in dataset:
        if EMBEDDINGS_FIELD in sample.field_names and sample[EMBEDDINGS_FIELD] is not None:
            embeddings_list.append(sample[EMBEDDINGS_FIELD])
            sample_ids_list.append(sample.id)
    
    if not embeddings_list:
        raise ValueError(f"No embeddings found in field '{EMBEDDINGS_FIELD}'")
    
    # Convert to numpy arrays
    embeddings_array = np.array(embeddings_list)
    sample_ids_array = np.array(sample_ids_list)
    
    print(f"Extracted {len(embeddings_list)} embeddings")
    print(f"Embedding shape: {embeddings_array.shape}")
    
    # Save to disk
    print(f"Saving embeddings to {EMBEDDINGS_FILE}...")
    np.save(EMBEDDINGS_FILE, embeddings_array)
    
    print(f"Saving sample IDs to {SAMPLE_IDS_FILE}...")
    np.save(SAMPLE_IDS_FILE, sample_ids_array)
    
    print("Done! Embeddings saved successfully.")
    print(f"  - Embeddings: {EMBEDDINGS_FILE}")
    print(f"  - Sample IDs: {SAMPLE_IDS_FILE}")


if __name__ == "__main__":
    main()

