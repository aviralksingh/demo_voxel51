import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.utils.transformers as fouhft
import transformers

print(f"list of datasets: {fo.list_datasets()}")

dataset = fo.load_dataset("coco-2017-validation")

transformers_model = transformers.AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
model_config = fouhft.FiftyOneTransformerConfig(
    {
        "model": transformers_model,
        "name_or_path":"facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
)
model = fouhft.FiftyOneTransformer(model_config)

dataset.compute_embeddings(model, embeddings_field="embeddings_dinov3")

viz = fob.compute_visualization(
    dataset,
    embeddings="embeddings_dinov3",
    brain_key="dino_dense_umap"
)
session = fo.launch_app(dataset, port=5151)
print(session.url)
