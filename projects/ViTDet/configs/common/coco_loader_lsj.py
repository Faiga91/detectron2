import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

from detectron2.data import DatasetMapper 

# Data using LSJ
image_size = 512
dataloader = model_zoo.get_config("common/data/coco.py").dataloader

# Custom mapper to remove masks
class CustomDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Modify the default DatasetMapper to remove mask annotations.
        """
        dataset_dict = super().__call__(dataset_dict)
        if "instances" in dataset_dict and dataset_dict["instances"].has("gt_masks"):
            dataset_dict["instances"].remove("gt_masks")  # Remove ground-truth masks
        return dataset_dict

# Update train mapper
dataloader.train.mapper = L(CustomDatasetMapper)(
    is_train=True,
    augmentations=[
        # Resize images to a fixed size
        L(T.Resize)(shape=(image_size, image_size)),
        # Add random horizontal flip
        L(T.RandomFlip)(horizontal=True),
    ],
    image_format="RGB",
    recompute_boxes=False,
)

# Update train batch size
dataloader.train.total_batch_size = 4

# Update test mapper
dataloader.test.mapper = L(CustomDatasetMapper)(
    is_train=False,
    augmentations=[
        # Resize images to a fixed size
        L(T.Resize)(shape=(image_size, image_size)),
    ],
    image_format="RGB",
)
dataloader.test.batch_size = 4
