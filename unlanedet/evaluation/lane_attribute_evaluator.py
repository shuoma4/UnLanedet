import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import logging


class LaneAttributeEvaluator:
    """Evaluator for lane attribute prediction.

    Evaluates:
    - Lane detection performance (TP, FP, FN, F1)
    - Lane category classification accuracy
    - Left-right attribute accuracy
    """

    def __init__(self, iou_threshold=0.5, width=30, metric='detection/f1'):
        """
        Args:
            iou_threshold: IoU threshold for matching
            width: Lane width for IoU calculation
            metric: Metric name to track for best checkpoint (default: 'detection/f1')
        """
        self.iou_threshold = iou_threshold
        self.width = width
        self.metric = metric
        self.logger = logging.getLogger(__name__)

        # Lane category mapping
        self.num_categories = 14

        # Left-right attribute mapping
        self.num_attributes = 4

        # Statistics
        self.reset() 

    def reset(self):
        """Reset all statistics."""
        self.tp = 0  # True positives for detection
        self.fp = 0  # False positives for detection
        self.fn = 0  # False negatives for detection

        # Category classification
        self.category_correct = 0
        self.category_total = 0
        self.category_confusion = np.zeros((self.num_categories, self.num_categories))

        # Attribute classification
        self.attribute_correct = 0
        self.attribute_total = 0
        self.attribute_confusion = np.zeros((self.num_attributes, self.num_attributes))

    def update(self, predictions, targets):
        """
        Update evaluation statistics with a batch of predictions.

        Args:
            predictions: List of prediction dictionaries, each containing:
                - 'lanes': list of lane points
                - 'category_logits': category predictions (optional)
                - 'attribute_logits': attribute predictions (optional)
                - 'cls_scores': confidence scores
            targets: List of target dictionaries, each containing:
                - 'lanes': list of lane points
                - 'lane_categories': category labels (optional)
                - 'lane_attributes': attribute labels (optional)
        """
        for pred, target in zip(predictions, targets):
            self._update_single(pred, target)

    def _update_single(self, pred, target):
        """Update statistics for a single image."""
        pred_lanes = pred.get('lanes', [])
        target_lanes = target.get('lanes', [])

        # Skip if no lanes
        if len(pred_lanes) == 0 and len(target_lanes) == 0:
            return

        if len(pred_lanes) == 0:
            self.fn += len(target_lanes)
            return

        if len(target_lanes) == 0:
            self.fp += len(pred_lanes)
            return

        # Match predictions to targets using IoU
        ious = self._compute_iou_matrix(pred_lanes, target_lanes)
        row_ind, col_ind = linear_sum_assignment(1 - ious)

        # Evaluate matches
        for i, j in zip(row_ind, col_ind):
            if ious[i, j] >= self.iou_threshold:
                # True positive
                self.tp += 1

                # Evaluate category prediction
                if 'category_logits' in pred and 'lane_categories' in target:
                    pred_category = pred['category_logits'][i].argmax().item()
                    true_category = target['lane_categories'][j]
                    self._update_category_stats(pred_category, true_category)

                # Evaluate attribute prediction
                if 'attribute_logits' in pred and 'lane_attributes' in target:
                    pred_attribute = pred['attribute_logits'][i].argmax().item()
                    true_attribute = target['lane_attributes'][j]
                    self._update_attribute_stats(pred_attribute, true_attribute)
            else:
                # False positive
                self.fp += 1

        # Count false negatives
        matched_targets = set(col_ind[ious[row_ind, col_ind] >= self.iou_threshold])
        unmatched_targets = set(range(len(target_lanes))) - matched_targets
        self.fn += len(unmatched_targets)

    def _compute_iou_matrix(self, pred_lanes, target_lanes):
        """Compute IoU matrix between predicted and target lanes."""
        ious = np.zeros((len(pred_lanes), len(target_lanes)))

        for i, pred_lane in enumerate(pred_lanes):
            pred_mask = self._lane_to_mask(pred_lane)
            for j, target_lane in enumerate(target_lanes):
                target_mask = self._lane_to_mask(target_lane)
                intersection = np.sum(pred_mask & target_mask)
                union = np.sum(pred_mask | target_mask)
                if union > 0:
                    ious[i, j] = intersection / union

        return ious

    def _lane_to_mask(self, lane_points):
        """Convert lane points to binary mask."""
        # Create a binary mask for the lane
        mask = np.zeros((self.height, self.width), dtype=bool)

        if len(lane_points) < 2:
            return mask

        # Draw line on mask
        for i in range(len(lane_points) - 1):
            x1, y1 = lane_points[i]
            x2, y2 = lane_points[i + 1]

            # Clip coordinates
            x1, x2 = np.clip([x1, x2], 0, self.width - 1).astype(int)
            y1, y2 = np.clip([y1, y2], 0, self.height - 1).astype(int)

            # Simple line drawing
            num_points = int(np.hypot(x2 - x1, y2 - y1)) + 1
            if num_points > 1:
                xs = np.linspace(x1, x2, num_points).astype(int)
                ys = np.linspace(y1, y2, num_points).astype(int)

                # Draw with width
                for x, y in zip(xs, ys):
                    half_width = self.width // 2
                    for dx in range(-half_width, half_width + 1):
                        for dy in range(-half_width, half_width + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                mask[ny, nx] = True

        return mask

    def _update_category_stats(self, pred_category, true_category):
        """Update category classification statistics."""
        self.category_total += 1
        if pred_category == true_category:
            self.category_correct += 1
        self.category_confusion[true_category, pred_category] += 1

    def _update_attribute_stats(self, pred_attribute, true_attribute):
        """Update attribute classification statistics."""
        self.attribute_total += 1
        if pred_attribute == true_attribute:
            self.attribute_correct += 1
        self.attribute_confusion[true_attribute, pred_attribute] += 1

    def get_results(self):
        """Get evaluation results."""
        # Detection metrics
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Category metrics
        category_accuracy = self.category_correct / self.category_total if self.category_total > 0 else 0
        category_precision = np.diag(self.category_confusion) / (
            self.category_confusion.sum(axis=0) + 1e-10)
        category_recall = np.diag(self.category_confusion) / (
            self.category_confusion.sum(axis=1) + 1e-10)
        category_f1 = 2 * category_precision * category_recall / (
            category_precision + category_recall + 1e-10)

        # Attribute metrics
        attribute_accuracy = self.attribute_correct / self.attribute_total if self.attribute_total > 0 else 0
        attribute_precision = np.diag(self.attribute_confusion) / (
            self.attribute_confusion.sum(axis=0) + 1e-10)
        attribute_recall = np.diag(self.attribute_confusion) / (
            self.attribute_confusion.sum(axis=1) + 1e-10)
        attribute_f1 = 2 * attribute_precision * attribute_recall / (
            attribute_precision + attribute_recall + 1e-10)

        results = {
            'detection': {
                'tp': self.tp,
                'fp': self.fp,
                'fn': self.fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            },
            'category': {
                'accuracy': category_accuracy,
                'precision': category_precision,
                'recall': category_recall,
                'f1': category_f1,
                'confusion_matrix': self.category_confusion,
            },
            'attribute': {
                'accuracy': attribute_accuracy,
                'precision': attribute_precision,
                'recall': attribute_recall,
                'f1': attribute_f1,
                'confusion_matrix': self.attribute_confusion,
            },
        }

        return results

    def print_results(self, results):
        """Print formatted results."""
        self.logger.info("=" * 80)
        self.logger.info("Lane Detection and Attribute Evaluation Results")
        self.logger.info("=" * 80)

        self.logger.info("\n--- Detection Metrics ---")
        self.logger.info(f"TP: {results['detection']['tp']}, "
                       f"FP: {results['detection']['fp']}, "
                       f"FN: {results['detection']['fn']}")
        self.logger.info(f"Precision: {results['detection']['precision']:.4f}")
        self.logger.info(f"Recall: {results['detection']['recall']:.4f}")
        self.logger.info(f"F1 Score: {results['detection']['f1']:.4f}")

        self.logger.info("\n--- Lane Category Classification ---")
        self.logger.info(f"Accuracy: {results['category']['accuracy']:.4f}")
        self.logger.info(f"Mean Precision: {results['category']['precision'].mean():.4f}")
        self.logger.info(f"Mean Recall: {results['category']['recall'].mean():.4f}")
        self.logger.info(f"Mean F1: {results['category']['f1'].mean():.4f}")

        self.logger.info("\n--- Left-Right Attribute Classification ---")
        self.logger.info(f"Accuracy: {results['attribute']['accuracy']:.4f}")
        self.logger.info(f"Mean Precision: {results['attribute']['precision'].mean():.4f}")
        self.logger.info(f"Mean Recall: {results['attribute']['recall'].mean():.4f}")
        self.logger.info(f"Mean F1: {results['attribute']['f1'].mean():.4f}")

        self.logger.info("=" * 80)

    def set_image_size(self, height, width):
        """Set image size for mask creation."""
        self.height = height
        self.width = width


class TemporalEvaluator:
    """Evaluator for temporal lane detection.

    Evaluates detection performance across video sequences considering
    temporal consistency.
    """

    def __init__(self, evaluator_cls=LaneAttributeEvaluator):
        self.evaluator_cls = evaluator_cls
        self.evaluator = None
        self.frame_results = []

    def reset(self):
        """Reset all statistics."""
        self.frame_results = []
        if self.evaluator is not None:
            self.evaluator.reset()

    def update(self, predictions, targets, frame_idx=None):
        """Update with frame predictions."""
        if self.evaluator is None:
            self.evaluator = self.evaluator_cls()

        self.evaluator.update(predictions, targets)

        # Store results for temporal analysis
        if frame_idx is not None:
            self.frame_results.append({
                'frame_idx': frame_idx,
                'predictions': predictions,
                'targets': targets,
            })

    def get_results(self):
        """Get evaluation results."""
        if self.evaluator is None:
            return {}

        results = self.evaluator.get_results()

        # Add temporal consistency metrics
        if len(self.frame_results) > 1:
            results['temporal'] = self._compute_temporal_consistency()

        return results

    def _compute_temporal_consistency(self):
        """Compute temporal consistency metrics."""
        # Track ID consistency across frames
        lane_id_changes = 0
        total_lanes = 0

        for i in range(len(self.frame_results) - 1):
            current_frame = self.frame_results[i]
            next_frame = self.frame_results[i + 1]

            # Compare matched lanes
            current_lanes = current_frame['predictions']
            next_lanes = next_frame['predictions']

            # Simple consistency check: number of detected lanes
            lane_count_diff = abs(len(current_lanes) - len(next_lanes))
            lane_id_changes += lane_count_diff
            total_lanes += len(current_lanes)

        consistency_score = 1.0 - (lane_id_changes / (total_lanes + 1e-10))

        return {
            'consistency_score': consistency_score,
            'lane_count_variance': np.std([len(f['predictions']) for f in self.frame_results]),
        }
