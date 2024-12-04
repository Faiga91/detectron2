import logging
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision.ops import box_iou

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class BinaryClassificationEvaluator:
    def __init__(self, iou_threshold=0.5, conf_threshold=0.5):
        """
        Initialize the evaluator with configurable thresholds.

        Args:
            iou_threshold (float): IoU threshold for matching predictions with ground truth.
            conf_threshold (float): Confidence threshold for predictions.
        """
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def compute_confusion_matrix(self, preds, targets):
        """
        Compute confusion matrix for binary classification (class 0 vs class 1).

        Args:
            preds (list of dicts): Each dict contains 'boxes' (Tensor[N, 4]), 'scores' (Tensor[N]),
                                    and 'labels' (Tensor[N]).
            targets (list of dicts): Each dict contains 'boxes' (Tensor[M, 4]) and 'labels' (Tensor[M]).

        Returns:
            ndarray: Confusion matrix (2x2 for binary classification).
        """
        y_true = []  # Ground truth labels per image (binary: 0 or 1)
        y_pred = []  # Predicted labels per image (binary: 0 or 1)

        for pred, target in zip(preds, targets):
            pred_boxes = pred["boxes"][pred["scores"] >= self.conf_threshold]
            target_boxes = target["boxes"]

            # Initialize per-image counters
            tp, fp, fn = 0, 0, 0
            matched = torch.zeros(len(target_boxes), dtype=torch.bool)

            # Match predicted boxes with ground truth boxes
            for pb in pred_boxes:
                ious = box_iou(pb.unsqueeze(0), target_boxes).squeeze(0)

                if ious.numel() > 0:
                    max_iou, max_idx = ious.max(0)
                    if max_iou >= self.iou_threshold and not matched[max_idx]:
                        tp += 1
                        matched[max_idx] = True
                    else:
                        fp += 1
                else:
                    fp += 1

            # Count unmatched ground truth boxes as false negatives
            fn += (~matched).sum().item()

            # Assign per-image labels
            if len(target_boxes) > 0:
                y_true.append(0)  # Ground truth is Polyp (class 0)
            else:
                y_true.append(1)  # Ground truth is Normal Mucosa (class 1)

            if tp > 0:
                y_pred.append(0)  # Predicted as Polyp (class 0)
            else:
                y_pred.append(1)  # Predicted as Normal Mucosa (class 1)

        # Validation: Ensure all cases are accounted for
        total_samples = len(preds)  # Each image contributes one sample
        if len(y_true) != total_samples or len(y_pred) != total_samples:
            raise ValueError(
                f"Mismatch in sample count: expected {total_samples}, got {len(y_true)}"
            )

        # Compute confusion matrix
        return confusion_matrix(y_true, y_pred, labels=[0, 1])

    def plot_confusion_matrix(
        self,
        cm,
        class_names,
        output_path="../results/classification_metrics/confusion_matrix.png",
    ):
        """
        Plot the confusion matrix.

        Args:
            cm (ndarray): Confusion matrix.
            class_names (list of str): Class names.
            output_path (str): Path to save the confusion matrix plot.
        """
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path)

    def calculate_metrics(self, cm):
        """
        Calculate classification metrics from the confusion matrix.

        Args:
            cm (ndarray): Confusion matrix.

        Returns:
            dict: Classification metrics (Accuracy, Recall, Precision, F1 Score, Specificity).
        """
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]

        # Metrics calculation
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            "Accuracy": accuracy,
            "Sensitivity (Recall)": recall,
            "Precision": precision,
            "F1 Score": f1_score,
            "Per-polyp Specificity": specificity,
        }
