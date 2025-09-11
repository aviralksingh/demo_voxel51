import fiftyone as fo
import fiftyone.zoo as foz

# You can load your own dataset
dataset = foz.load_zoo_dataset(
    "https://github.com/voxel51/coco-2017",
    split="validation",
    dataset_name="coco-2017-validation",
)

dataset = fo.load_dataset("coco-2017-validation")

# Visualize
session = fo.launch_app(dataset)
session.wait()  # keeps the app running
