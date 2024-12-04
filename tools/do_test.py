import logging
import json
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_on_dataset

from detectron2.data import DatasetCatalog, DatasetFromList, DatasetMapper

from detectron2.evaluation import print_csv_format

from detectron2.structures import Instances, Boxes
import torch
import cv2

from binary_classification_evaluator import BinaryClassificationEvaluator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

## OUS-20220203
test_json_path = "/dataset/ous-20220203-copy/annotations/test_updated.json"
image_dir = "/dataset/ous-20220203-copy/images/test"

## Hyper-Kvasir
# test_json_path = "/dataset/hyper-kvasir/test-COCO-annotations.json"
# image_dir = "/dataset/hyper-kvasir/test"

config_file = "../projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
weights_path = "output/model_final.pth"


def custom_mapper(dataset_dict, augmentations):
    """
    Custom mapper to create instances for each dataset entry with augmentations.

    Args:
        dataset_dict (dict): A single dataset entry.
        augmentations (list): A list of augmentation transforms.

    Returns:
        dict: Updated dataset entry with `instances`.
    """
    dataset_dict = dataset_dict.copy()  # Avoid modifying the original dictionary

    # Read the image
    image = cv2.imread(dataset_dict["file_name"])
    if image is None:
        raise FileNotFoundError(f"Image not found: {dataset_dict['file_name']}")
    image = image[:, :, ::-1]  # Convert BGR to RGB

    # Apply augmentations
    from detectron2.data import transforms as T

    aug_input = T.AugInput(image)
    transforms = T.AugmentationList(augmentations)(aug_input)
    image = aug_input.image

    # Update the dataset with the transformed image
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    # Process annotations if available
    if "annotations" in dataset_dict:
        annotations = dataset_dict.pop("annotations")

        # Apply the same transforms to annotations (e.g., bounding boxes)
        for anno in annotations:
            bbox = anno["bbox"]
            # bbox format is XYWH, convert to XYXY for transformation
            bbox_xyxy = [
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3],
            ]
            bbox_xyxy = transforms.apply_box([bbox_xyxy])[0]
            # Convert back to XYWH format
            anno["bbox"] = [
                bbox_xyxy[0],
                bbox_xyxy[1],
                bbox_xyxy[2] - bbox_xyxy[0],
                bbox_xyxy[3] - bbox_xyxy[1],
            ]

        # Create Instances for the transformed annotations
        instances = Instances((image.shape[0], image.shape[1]))
        instances.gt_boxes = Boxes([anno["bbox"] for anno in annotations])
        instances.gt_classes = torch.tensor(
            [anno["category_id"] for anno in annotations], dtype=torch.int64
        )
        dataset_dict["instances"] = instances

    return dataset_dict


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
mapped_dataset = list(map(lambda d: custom_mapper(d, augmentations), wrapped_dataset))

cfg.dataloader.test = {
    "_target_": "detectron2.data.build_detection_test_loader",
    "dataset": mapped_dataset,  # Use the preprocessed dataset
    "mapper": None,  # Mapper is already applied
    "num_workers": 0,
    "batch_size": 4,
}

cfg.dataloader.evaluator = {
    "_target_": "detectron2.evaluation.COCOEvaluator",
    "dataset_name": "my_test_dataset",
    "output_dir": "eval_results",
}

cfg.dataloader.test.batch_size = 2
cfg.train.init_checkpoint = weights_path
cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
cfg.model.roi_heads.num_classes = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the model
model = instantiate(cfg.model).to(device)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
model.eval()


def do_test(cfg, model):
    """
    Run inference on the test dataset and compute evaluation metrics.

    Args:
        cfg: Configuration object with dataloader and evaluator settings.
        model: Trained model for inference.

    Returns:
        dict: Evaluation results and classification metrics.
    """

    if "evaluator" in cfg.dataloader:
        # Run inference
        results = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(results)

    # Load saved predictions from coco_instances_results.json
    logger.info("Loading predictions from coco_instances_results.json...")
    predictions_path = (
        f"{cfg.dataloader.evaluator.output_dir}/coco_instances_results.json"
    )

    evaluator = BinaryClassificationEvaluator()

    confusion_matrix = None
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

        # Extract predictions and targets for binary classification evaluation
        # logger.info("Extracting predictions and ground truth for binary evaluation...")
        # dataloader = instantiate(cfg.dataloader.test)
        # all_preds = []
        # all_targets = []

        for dataset_dict in dataset_dicts:  # Access the dataset directly
            image_id = dataset_dict["image_id"]

            # Match predictions to the current image
            preds = [pred for pred in predictions if pred["image_id"] == image_id]
            pred_boxes = torch.tensor(
                [pred["bbox"] for pred in preds], dtype=torch.float32
            )
            pred_scores = torch.tensor(
                [pred["score"] for pred in preds], dtype=torch.float32
            )
            pred_labels = torch.tensor(
                [pred["category_id"] for pred in preds], dtype=torch.int64
            )

            # Get ground truth
            gt_boxes = torch.tensor(
                [anno["bbox"] for anno in dataset_dict["annotations"]],
                dtype=torch.float32,
            )
            gt_labels = torch.tensor(
                [anno["category_id"] for anno in dataset_dict["annotations"]],
                dtype=torch.int64,
            )

            batch_preds = [
                {"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}
            ]
            batch_targets = [{"boxes": gt_boxes, "labels": gt_labels}]

            batch_cm = evaluator.compute_confusion_matrix(batch_preds, batch_targets)
            if confusion_matrix is None:
                confusion_matrix = batch_cm
            else:
                confusion_matrix += batch_cm

            del (
                batch_preds,
                batch_targets,
                pred_scores,
                pred_labels,
                gt_labels,
                gt_boxes,
            )

        metrics = evaluator.calculate_metrics(confusion_matrix)
        evaluator.plot_confusion_matrix(
            confusion_matrix,
            class_names=["Polyp", "Normal Mucosa"],
            output_path="eval_results/confusion_matrix.png",
        )

        # Return results
        return {
            # "detection_results": results,
            "binary_classification_metrics": metrics
        }


results = do_test(cfg, model)
for key, value in results.items():
    if key == "binary_classification_metrics":
        logger.info(f"{key}:")
        for k, v in value.items():
            logger.info(f"{key}.{k}: {v}")
