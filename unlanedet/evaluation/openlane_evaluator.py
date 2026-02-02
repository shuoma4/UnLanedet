import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import json
import shutil
import subprocess
from scipy.optimize import linear_sum_assignment
from .evaluator import DatasetEvaluator
from ..model.LLANet.line_iou import line_iou


def lane_nms(predictions, scores, nms_overlap_thresh=50.0, top_k=24, img_w=1920):
    """
    NMS function with coordinate scaling fix.
    """
    # Memory: NMS threshold must NOT be normalized
    # if nms_overlap_thresh > 1.0:
    #    nms_overlap_thresh /= 100.0

    keep_index = []
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    sorted_indices = sorted_indices[:top_k]

    lane_points = predictions[:, 5:] * img_w

    while sorted_indices.size(0) > 0:
        idx = sorted_indices[0]
        keep_index.append(idx.item())

        if sorted_indices.size(0) == 1:
            break

        sorted_indices = sorted_indices[1:]

        curr_lane = lane_points[idx].unsqueeze(0)
        rest_lanes = lane_points[sorted_indices]

        # 计算 IoU
        ious = line_iou(curr_lane, rest_lanes, img_w=img_w, length=30)

        # NMS 过滤
        mask = ious < nms_overlap_thresh
        sorted_indices = sorted_indices[mask.view(-1)]

    return keep_index


class OpenLaneEvaluator(DatasetEvaluator):
    """
    OpenLane Official Evaluator with Incremental Saving.

    Features:
    1. Decodes model predictions to original image coordinates (1920x1280).
    2. Incrementally exports predictions to OpenLane-format JSON files.
    3. Calls the official C++ binary (`evaluate`) to calculate F-Score / IoU metrics.
    4. Calculates internal Python metrics for Attribute & Category accuracy.
    """

    def __init__(
        self,
        cfg,
        evaluate_bin_path=None,
        output_dir=None,
        iou_threshold=0.5,
        width=30,
        metric="OpenLane/F1",
    ):
        """
        Args:
            cfg: Config object (Required)
            evaluate_bin_path (str): Path to the compiled C++ evaluation tool.
            output_dir (str): Directory to save JSON results and logs.
            iou_threshold (float): IoU threshold for internal attribute matching and official tool.
            width (int): Lane width for IoU calculation (passed to official tool).
            metric (str): Primary metric name (for Best Checkpoint selection).
        """
        self.cfg = cfg
        self.evaluate_bin_path = evaluate_bin_path
        self.iou_threshold = iou_threshold
        self.width = width
        self.metric = metric  # Renamed back to metric to match original logic if needed, but primary_metric_name was better. Wait, the old code used primary_metric_name?
        self.logger = logging.getLogger(__name__)

        # 1. Output Directory Setup
        if output_dir is None:
            if hasattr(cfg, "train") and hasattr(cfg.train, "output_dir"):
                self.output_dir = os.path.join(cfg.train.output_dir, "eval_results")
            else:
                self.output_dir = "./output/eval_results"
        else:
            self.output_dir = output_dir

        self.result_dir = self.output_dir
        self.test_list_path = os.path.join(self.output_dir, "test_list.txt")

        # Visualization directories for annotated images
        self.visualization_dir = os.path.join(self.output_dir, "visualization")
        self.annotated_images_dir = os.path.join(
            self.visualization_dir, "annotated_images"
        )
        self.script_path = getattr(
            cfg,
            "parse_evaluate_result_script",
            os.path.join(
                os.path.dirname(__file__), "../../../tools/read_open2d_csv_results.py"
            ),
        )

        # 2. Category Mapping (Model 0-14 -> OpenLane Official IDs)
        self.id_to_openlane_cat = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
            13: 21,  # Left Curbside -> Right Curbside (Swap to fix mismatch)
            14: 20,  # Right Curbside -> Left Curbside (Swap to fix mismatch)
        }

        # 3. Stats Initialization
        self.num_categories = getattr(cfg, "num_lane_categories", 15)
        self.num_attributes = getattr(cfg, "num_lr_attributes", 4)
        self._created_dirs = set()

        # Configuration for visualization
        self.generate_visualization = getattr(cfg, "generate_visualization", False)
        self.reset()

    def reset(self):
        # Internal Python Stats
        self.category_correct = 0
        self.category_total = 0
        self.attribute_correct = 0
        self.attribute_total = 0
        self._created_dirs.clear()

        # Clean up old results directory
        if os.path.exists(self.output_dir):
            try:
                if os.path.exists(self.result_dir):
                    shutil.rmtree(self.result_dir)
                if os.path.exists(self.test_list_path):
                    os.remove(self.test_list_path)
                if os.path.exists(self.visualization_dir):
                    shutil.rmtree(self.visualization_dir)
            except Exception as e:
                self.logger.warning(f"Failed to clean output dir: {e}")

        os.makedirs(self.result_dir, exist_ok=True)

        # Create visualization directories if needed
        if self.generate_visualization:
            os.makedirs(self.annotated_images_dir, exist_ok=True)
            self.logger.info(
                f"[OpenLaneEvaluator] Visualization enabled. Annotated images will be saved to: {self.annotated_images_dir}"
            )

    def process(self, inputs, outputs):
        # Extract batch-level category and attribute logits if available
        cat_logits_batch = None
        attr_logits_batch = None

        if isinstance(outputs, dict):
            if "category" in outputs:
                cat_logits_batch = outputs["category"]
            if "attribute" in outputs:
                attr_logits_batch = outputs["attribute"]
            if "lane_lines" in outputs:
                outputs = outputs["lane_lines"]

        batch_size = len(outputs)

        for i in range(batch_size):
            output_data = outputs[i]

            # --- 1. 解析 Meta 信息 ---
            original_rel_path = None
            if "img_name" in inputs:
                names = inputs["img_name"]
                if isinstance(names, list) and i < len(names):
                    original_rel_path = names[i]
                elif (
                    hasattr(names, "data")
                    and isinstance(names.data, list)
                    and i < len(names.data)
                ):
                    original_rel_path = names.data[i]

            if not original_rel_path:
                raw_meta = inputs.get("meta", inputs.get("img_metas", [{}]))
                if hasattr(raw_meta, "data"):
                    raw_meta = raw_meta.data
                while (
                    isinstance(raw_meta, list)
                    and len(raw_meta) > 0
                    and isinstance(raw_meta[0], list)
                ):
                    raw_meta = raw_meta[0]

                meta = {}
                if isinstance(raw_meta, list) and i < len(raw_meta):
                    meta = raw_meta[i]
                elif isinstance(raw_meta, dict):
                    meta = {
                        k: (
                            v[i]
                            if isinstance(v, (list, torch.Tensor)) and len(v) > i
                            else v
                        )
                        for k, v in raw_meta.items()
                    }

                for key in ["img_name", "filename", "img_path", "file_name"]:
                    if isinstance(meta, dict) and meta.get(key):
                        original_rel_path = meta[key]
                        break

            if not original_rel_path:
                self.logger.warning(
                    f"[WARNING] Meta content (partial): {str(inputs.keys())}"
                )
                continue

            # FIX: Normalize path prefix to match annotation directory structure (lane3d_300/validation)
            if "training_resized_800_320/validation" in original_rel_path:
                original_rel_path = original_rel_path.replace(
                    "training_resized_800_320/validation", "validation"
                )
            elif "validation_resized_800_320/" in original_rel_path:
                original_rel_path = original_rel_path.replace(
                    "validation_resized_800_320/", "validation/"
                )
            elif original_rel_path.startswith("validation/"):
                pass  # Already correct

            # Use full relative path (including validation/) for saving to match GT structure
            save_rel_path = original_rel_path

            # --- 2. 准备预测数据 ---
            if isinstance(output_data, dict):
                pred_lines = output_data.get("lane_lines", [])
            else:
                pred_lines = output_data

            if not torch.is_tensor(pred_lines):
                pred_lines = torch.tensor(pred_lines)
            pred_lines = pred_lines.detach().cpu()

            if len(pred_lines) == 0:
                self._save_empty_json(save_rel_path, original_rel_path)
                continue

            # 处理维度：如果 pred_lines 是 3D tensor [batch, num_priors, dim]，取第一个样本
            if pred_lines.dim() == 3:
                pred_lines = pred_lines[0]  # [num_priors, dim]

            # --- 3. Softmax 处理 (关键步骤) ---
            # 原始输出前两列是 Logits，必须转概率
            logits = pred_lines[:, :2]
            probs = F.softmax(logits, dim=1)
            scores = probs[:, 1]

            conf_threshold = self.cfg.test_parameters.conf_threshold
            valid_mask = scores > conf_threshold
            valid_preds = pred_lines[valid_mask]
            valid_scores = scores[valid_mask]

            if valid_mask.sum() == 0 and len(scores) > 0:
                self.logger.error(f"No preds' confidence > {conf_threshold}")
                self._save_empty_json(save_rel_path, original_rel_path)
                continue

            nms_preds = valid_preds.clone()
            keep_indices = lane_nms(
                nms_preds,
                valid_scores,
                nms_overlap_thresh=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.test_parameters.nms_topk,
                img_w=self.cfg.img_w,  # 传入 img_w 用于反归一化
            )
            # 获取 NMS 后的最终预测
            final_preds = valid_preds[keep_indices]
            # --- 5. 处理类别和属性 ---
            valid_cats = np.zeros(len(final_preds), dtype=int)
            valid_attrs = np.zeros(len(final_preds), dtype=int)
            # Handle Category
            current_cat_logits = None
            if cat_logits_batch is not None:
                current_cat_logits = cat_logits_batch[i]
            elif isinstance(output_data, dict) and "category" in output_data:
                current_cat_logits = output_data["category"]
            if current_cat_logits is not None:
                if torch.is_tensor(current_cat_logits):
                    current_cat_logits = current_cat_logits.detach().cpu()
                    # Filter by valid_mask
                    if current_cat_logits.shape[0] == valid_mask.shape[0]:
                        cat_logits_valid = current_cat_logits[valid_mask]
                        # Filter by NMS keep_indices
                        cat_logits_final = cat_logits_valid[keep_indices]
                        valid_cats = torch.argmax(cat_logits_final, dim=1).numpy()
                    else:
                        self.logger.warning(
                            f"Shape mismatch for category: {current_cat_logits.shape} vs mask {valid_mask.shape}"
                        )

            # Handle Attribute
            current_attr_logits = None
            if attr_logits_batch is not None:
                current_attr_logits = attr_logits_batch[i]
            elif isinstance(output_data, dict) and "attribute" in output_data:
                current_attr_logits = output_data["attribute"]

            if current_attr_logits is not None:
                if torch.is_tensor(current_attr_logits):
                    current_attr_logits = current_attr_logits.detach().cpu()
                    # Filter by valid_mask
                    if current_attr_logits.shape[0] == valid_mask.shape[0]:
                        attr_logits_valid = current_attr_logits[valid_mask]
                        # Filter by NMS keep_indices
                        attr_logits_final = attr_logits_valid[keep_indices]
                        valid_attrs = torch.argmax(attr_logits_final, dim=1).numpy()
                    else:
                        self.logger.warning(
                            f"Shape mismatch for attribute: {current_attr_logits.shape} vs mask {valid_mask.shape}"
                        )

            # --- 6. 解码坐标并保存 ---
            # decode_lanes 内部会乘以 ori_img_w，所以这里传入 normalized 的 final_preds 即可
            decoded_lanes = self.decode_lanes(
                final_preds.numpy(),
                self.cfg.img_w,
                self.cfg.img_h,
                self.cfg.ori_img_w,
                self.cfg.ori_img_h,
                self.cfg.num_points,
            )

            lane_lines_json = []
            for j, lane_points in enumerate(decoded_lanes):
                if len(lane_points) < 2:
                    continue
                model_cat_id = int(valid_cats[j])
                official_cat_id = self.id_to_openlane_cat.get(model_cat_id, 0)

                model_attr_id = int(valid_attrs[j])
                official_attr_id = model_attr_id

                # Convert to [ [x1, x2, ...], [y1, y2, ...] ] format for OpenLane C++ evaluator
                xs = []
                ys = []
                for k in range(len(lane_points)):
                    xs.append(float(lane_points[k][0]))
                    ys.append(float(lane_points[k][1]))

                line_2d = [xs, ys]

                lane_obj = {
                    "category": official_cat_id,
                    "attribute": official_attr_id,
                    "uv": line_2d,
                }
                lane_lines_json.append(lane_obj)

            self._write_json(
                save_rel_path,
                {
                    "file_path": original_rel_path,
                    "lane_lines": lane_lines_json,
                },
            )

        return []

    def _write_json(self, rel_path, content):
        json_rel_path = rel_path.replace(".jpg", ".json")
        save_path = os.path.join(self.result_dir, json_rel_path)
        dir_name = os.path.dirname(save_path)
        if dir_name not in self._created_dirs:
            os.makedirs(dir_name, exist_ok=True)
            self._created_dirs.add(dir_name)
            # Only print when the first directory is created (start of generation)
            if len(self._created_dirs) == 1:
                self.logger.info(
                    f"[OpenLaneEvaluator] Generating validation predictions to: {self.result_dir}"
                )
        with open(save_path, "w") as f:
            json.dump(content, f)

        # Generate visualization if enabled
        if self.generate_visualization:
            self._generate_visualization(rel_path, content)

    def _generate_visualization(self, rel_path, json_content):
        """
        Generate annotated images using tools/read_open2d_csv_results.py
        """
        try:
            # Extract image filename from rel_path
            img_filename = os.path.basename(rel_path).replace(".json", ".jpg")

            # Path to the read_open2d_csv_results.py script
            script_path = self.script_path

            if not os.path.exists(script_path):
                self.logger.warning(f"Visualization script not found: {script_path}")
                return

            # Prepare command to generate visualization
            # The script expects to read from csv_results/, so we need to run evaluation first
            # For now, we'll create a placeholder to indicate this should be called after evaluation
            vis_cmd = [
                "python",
                script_path,
                "--csv-folder",
                os.path.join(self.output_dir, "csv_results"),
                "--visualize",
                "--output-dir",
                self.annotated_images_dir,
            ]

            # Store visualization command for later execution
            if not hasattr(self, "_vis_commands"):
                self._vis_commands = []
            self._vis_commands.append(
                {
                    "cmd": vis_cmd,
                    "img_filename": img_filename,
                    "json_content": json_content,
                }
            )

        except Exception as e:
            self.logger.warning(f"Failed to prepare visualization for {rel_path}: {e}")

    def _save_empty_json(self, rel_path, full_path):
        self._write_json(rel_path, {"file_path": full_path, "lane_lines": []})

    def evaluate(self, predictions=None):
        self.logger.info(
            f"[OpenLaneEvaluator] Inference finished. Generating test list..."
        )

        all_json_files = []
        for root, dirs, files in os.walk(self.result_dir):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "r") as f:
                            content = json.load(f)
                            img_path = None
                            if "file_path" in content:
                                img_path = content["file_path"]

                            if img_path:
                                all_json_files.append(img_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to read {full_path}: {e}")

        if len(all_json_files) == 0:
            self.logger.warning("No JSON results found! Evaluation skipped.")
            return {}

        self.logger.info(
            f"[OpenLaneEvaluator] Generating test_list.txt with {len(all_json_files)} images..."
        )
        with open(self.test_list_path, "w") as f:
            for path in all_json_files:
                f.write(path + "\n")
        self.logger.info(
            f"[OpenLaneEvaluator] test_list.txt generated at: {self.test_list_path}"
        )

        # Official C++ Tool
        metrics = {}
        if not os.path.exists(self.evaluate_bin_path):
            self.logger.error(
                f"Cannot find official evaluate binary at: {self.evaluate_bin_path}"
            )
        else:
            metrics = self._run_official_evaluation()

            # Generate visualizations after evaluation if enabled
            if (
                self.generate_visualization
                and hasattr(self, "_vis_commands")
                and self._vis_commands
            ):
                self._execute_visualizations()

        # Merge Internal Python Metrics
        local_metrics = self._get_internal_metrics()
        metrics.update(local_metrics)

        self.logger.info("=" * 40)
        self.logger.info(f"OpenLane Evaluation Results:")
        for k, v in metrics.items():
            self.logger.info(f"{k}: {v:.4f}")
        self.logger.info("=" * 40)

        # Log visualization results
        if self.generate_visualization:
            self.logger.info(
                f"[OpenLaneEvaluator] Visualizations saved to: {self.annotated_images_dir}"
            )

        return metrics

    def _execute_visualizations(self):
        """
        Execute visualization generation after evaluation completes
        """
        try:
            # First run the official evaluation to generate csv_results/
            self.logger.info(
                "[OpenLaneEvaluator] Running evaluation to generate CSV results for visualization..."
            )

            # The CSV results should already be generated by _run_official_evaluation
            csv_results_dir = os.path.join(self.output_dir, "csv_results")

            if not os.path.exists(csv_results_dir):
                self.logger.warning(
                    f"CSV results directory not found: {csv_results_dir}"
                )
                return

            # Generate visualizations using read_open2d_csv_results.py
            script_path = self.script_path

            if not os.path.exists(script_path):
                self.logger.warning(f"Visualization script not found: {script_path}")
                return

            # Create visualization command
            vis_cmd = [
                "python",
                script_path,
                "--csv-folder",
                csv_results_dir,
                "--plot",
                os.path.join(self.visualization_dir, "performance.png"),
                "--excel",
                os.path.join(self.visualization_dir, "evaluation_results.xlsx"),
            ]

            self.logger.info(
                f"[OpenLaneEvaluator] Generating evaluation plots and Excel report..."
            )

            # Execute visualization command
            result = subprocess.run(
                vis_cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                self.logger.info(
                    f"[OpenLaneEvaluator] Evaluation visualization completed successfully"
                )
            else:
                self.logger.warning(
                    f"[OpenLaneEvaluator] Visualization generation had issues: {result.stderr}"
                )

        except Exception as e:
            self.logger.warning(
                f"[OpenLaneEvaluator] Failed to execute visualizations: {e}"
            )

    def _run_official_evaluation(self):
        lane_anno_dir = getattr(self.cfg, "lane_anno_dir", "lane3d_300/")
        # Handle both string and Path types for data_root
        data_root = self.cfg.data_root
        if hasattr(data_root, "as_posix"):
            data_root = data_root.as_posix()
        # Remove trailing slashes and let os.path.join handle the path properly
        dataset_dir = (
            os.path.join(data_root.rstrip("/"), lane_anno_dir.rstrip("/")) + "/"
        )
        image_dir = data_root.rstrip("/") + "/"
        result_dir_root = self.output_dir
        output_file = os.path.join(self.output_dir, "eval_res/")

        # Setup environment variables for OpenCV library path
        env = os.environ.copy()
        opencv_path = getattr(self.cfg, "opencv_path", None)
        if opencv_path:
            env["LD_LIBRARY_PATH"] = opencv_path
            self.logger.info(
                f"[OpenLaneEvaluator] Setting LD_LIBRARY_PATH={opencv_path}"
            )

        cmd = [
            str(self.evaluate_bin_path),
            "-a",
            dataset_dir,
            "-d",
            result_dir_root,
            "-i",
            image_dir,
            "-l",
            self.test_list_path,
            "-w",
            str(self.width),
            "-t",
            str(self.iou_threshold),
            "-o",
            output_file,
            "-j",
            str(4),
        ]
        if self.generate_visualization:
            cmd.append("-v")  # 保存可视化图片

        metrics = {}
        try:
            self.logger.info(
                f"[OpenLaneEvaluator] Running OpenLane official evaluation tool..."
            )
            self.logger.info(f"  Command: {' '.join(cmd)}")
            self.logger.info(
                f"  Parameters: width={self.width}, iou_threshold={self.iou_threshold}"
            )
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3000, env=env
            )

            # Log stderr if there's any error output
            if result.stderr:
                self.logger.error(
                    f"[OpenLaneEvaluator] Evaluation tool stderr: {result.stderr}"
                )

            # Log stdout for debugging
            if result.stdout:
                self.logger.debug(
                    f"[OpenLaneEvaluator] Evaluation tool stdout: {result.stdout}"
                )

            if result.stdout:
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if line.startswith("F-measure"):
                        metrics["OpenLane/F1"] = float(line.split(":")[1].strip())
                    elif line.startswith("Precision"):
                        metrics["OpenLane/Precision"] = float(
                            line.split(":")[1].strip()
                        )
                    elif line.startswith("Recall"):
                        metrics["OpenLane/Recall"] = float(line.split(":")[1].strip())
            else:
                self.logger.warning(
                    "[OpenLaneEvaluator] No stdout from evaluation tool"
                )
        except subprocess.TimeoutExpired:
            self.logger.error(
                "[OpenLaneEvaluator] Evaluation tool timed out after 300 seconds"
            )
        except Exception as e:
            self.logger.error(
                f"[OpenLaneEvaluator] Failed to execute evaluate binary: {e}"
            )
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
        return metrics

    def _update_internal_metrics(
        self, pred_lanes, pred_cats, pred_attrs, gt_lanes, gt_cats, gt_attrs
    ):
        if len(pred_lanes) == 0 or len(gt_lanes) == 0:
            return

        num_pred = len(pred_lanes)
        num_gt = len(gt_lanes)
        cost_matrix = np.zeros((num_pred, num_gt))

        for i in range(num_pred):
            for j in range(num_gt):
                p_lane = np.array(pred_lanes[i])
                g_lane = np.array(gt_lanes[j])
                d_start = np.linalg.norm(p_lane[0] - g_lane[0])
                d_end = np.linalg.norm(p_lane[-1] - g_lane[-1])
                cost_matrix[i, j] = d_start + d_end

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        MATCH_DIST_THRESH = 150.0

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < MATCH_DIST_THRESH:
                self.category_total += 1
                self.attribute_total += 1
                if int(pred_cats[i]) == int(gt_cats[j]):
                    self.category_correct += 1
                if int(pred_attrs[i]) == int(gt_attrs[j]):
                    self.attribute_correct += 1

    def _get_internal_metrics(self):
        cat_acc = (
            self.category_correct / self.category_total
            if self.category_total > 0
            else 0
        )
        attr_acc = (
            self.attribute_correct / self.attribute_total
            if self.attribute_total > 0
            else 0
        )
        return {"Internal/Category_Acc": cat_acc, "Internal/Attribute_Acc": attr_acc}

    def decode_lanes(self, preds, img_w, img_h, ori_img_w, ori_img_h, num_points):
        """
        将模型预测的归一化坐标解码回原始图像尺寸 (1920x1280)
        """
        decoded = []
        cut_height = self.cfg.cut_height
        strip_size = img_h / (num_points - 1)  # 320 / 71 ≈ 4.507
        orig_crop_h = ori_img_h - cut_height  # 1280 - 270 = 1010

        for lane in preds:
            # lane indices: 0:score, 1:cat, 2:start_y, 3:start_x, 4:length, 5..76:offsets
            start_y = lane[2].item() if torch.is_tensor(lane) else lane[2]
            length = lane[5].item() if torch.is_tensor(lane) else lane[5]
            real_start_y = 1.0 - start_y
            y_start_orig = real_start_y * orig_crop_h + cut_height
            y_end_orig = (real_start_y - length) * orig_crop_h + cut_height
            xs = lane[6:].detach().cpu().numpy() if torch.is_tensor(lane) else lane[6:]
            xs = xs[::-1]
            lane_points = []
            for i, x in enumerate(xs):
                y_train = i * strip_size
                y_orig = (y_train / img_h) * orig_crop_h + cut_height
                x_orig = x * ori_img_w
                if y_orig < y_end_orig - 2 or y_orig > y_start_orig + 2:
                    continue
                x_orig = x * ori_img_w
                if x_orig < 0 or x_orig > ori_img_w:
                    continue
                lane_points.append((x_orig, y_orig))
            lane_points = lane_points[::-1]
            decoded.append(lane_points)
        return decoded
