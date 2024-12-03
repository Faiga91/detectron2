import logging

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_on_dataset

from detectron2.data import DatasetCatalog, DatasetFromList, DatasetMapper

from detectron2.evaluation import print_csv_format

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

test_json_path = "/dataset/hyper-kvasir/test-COCO-annotations.json"
image_dir = "/dataset/hyper-kvasir/test"
config_file = "../projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
weights_path = "output/model_final.pth"

# Register the dataset
register_coco_instances("my_test_dataset", {}, test_json_path, image_dir)

dataset_dicts = DatasetCatalog.get("my_test_dataset")
logger.debug(dataset_dicts[:5])


# Load configuration and model
cfg = LazyConfig.load(config_file)

augmentations = [instantiate(aug) for aug in cfg.dataloader.test.mapper.augmentations]

wrapped_dataset = DatasetFromList(dataset_dicts, copy=False)

mapper = DatasetMapper(
    is_train=False,
    augmentations=augmentations,
    image_format="BGR",  # Ensure your image format matches the dataset
)
mapped_dataset = list(map(mapper, wrapped_dataset))

cfg.dataloader.test = {
    "_target_": "detectron2.data.build_detection_test_loader",
    "dataset": mapped_dataset,  # Use the preprocessed dataset
    "mapper": None,  # Mapper is already applied
    "num_workers": 4,
    "batch_size": 2,
}

cfg.dataloader.evaluator = {
    "_target_": "detectron2.evaluation.COCOEvaluator",
    "dataset_name": "my_test_dataset",
    "output_dir": "eval_results",
}

cfg.train.init_checkpoint = weights_path

# Initialize and load the model
model = instantiate(cfg.model).to("cuda")
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
model.eval()


# Run the test function
def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


results = do_test(cfg, model)
logger.info("Results: %s", results)
