import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.llanet.priors import CATEGORY_WEIGHT
from unlanedet.config import instantiate
from unlanedet.layers.ops import nms
from unlanedet.model.module.core.lane import Lane
from unlanedet.utils.detailed_loss_logger import DetailedLossLogger

from ..module.head.plaindecoder import PlainDecoder
from ..module.losses.focal_loss import FocalLoss
from .dynamic_assign import assign
from .line_iou import line_iou
from .roi_gather import LinearModule, ROIGather


class LLANetHead(nn.Module):
    def __init__(
        self,
        num_points=72,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_priors=192,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        cfg=None,
    ):
        super(LLANetHead, self).__init__()
        if cfg is None:
            raise ValueError('cfg must be provided')
        self.cfg = cfg
        self.decoder = instantiate(cfg.decoder)
        self.enable_category = cfg.enable_category
        self.enable_attribute = cfg.enable_attribute
        self.num_lane_categories = cfg.num_lane_categories
        self.num_lr_attributes = cfg.num_lr_attributes
        self.max_lanes = cfg.max_lanes
        self.scale_factor = cfg.scale_factor
        self.__init_detailed_logger()
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.num_fc = num_fc

        # Dynamic weight scheduling parameters
        self.epoch_per_iter = cfg.epoch_per_iter
        self.warmup_epochs = cfg.warmup_epochs
        self.start_category_loss_weight = cfg.start_category_loss_weight
        self.category_loss_weight = cfg.category_loss_weight
        self.start_attribute_loss_weight = cfg.start_attribute_loss_weight
        self.attribute_loss_weight = cfg.attribute_loss_weight
        self.logger = logging.getLogger(__name__)
        self.prior_feat_channels = prior_feat_channels

        # Buffers
        self.__init_buffers()
        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer('priors', init_priors)
        self.register_buffer('priors_on_featmap', priors_on_featmap)

        # seg module
        self.seg_conv = nn.Conv2d(self.prior_feat_channels * self.refine_layers, self.prior_feat_channels, 1)
        self.seg_decoder = PlainDecoder(cfg)

        self.roi_gather = ROIGather(
            in_channels=self.prior_feat_channels,
            num_priors=self.num_priors,
            sample_points=self.sample_points,
            fc_hidden_dim=self.fc_hidden_dim,
            refine_layers=self.refine_layers,
            mid_channels=self.fc_hidden_dim,
        )

        # init output layers
        self.__init_reg_layers()
        self.__init_cls_layers()
        self.__init_category_layers()
        self.__init_attribute_layers()

        # init loss function
        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label, weight=weights)
        if self.enable_category:
            self.category_criterion = torch.nn.NLLLoss(
                weight=torch.tensor(CATEGORY_WEIGHT, dtype=torch.float32),
                reduction='sum',
            )
            self.logger.info(f'category loss weights: {CATEGORY_WEIGHT}')

        if self.enable_attribute:
            self.attribute_criterion = torch.nn.NLLLoss(reduction='sum')

        self.init_weights()

    def __init_detailed_logger(self):
        self.detailed_loss_logger = self.cfg.get('detailed_loss_logger', None)
        if self.detailed_loss_logger is None and self.cfg.get('detailed_loss_logger_config') is not None:
            conf = self.cfg.get('detailed_loss_logger_config')
            self.detailed_loss_logger = DetailedLossLogger(output_dir=conf.output_dir, filename=conf.filename)

    def __init_reg_layers(self):
        reg_modules = []
        for _ in range(self.num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.reg_layers = nn.Linear(self.fc_hidden_dim, self.n_offsets + 1 + 2 + 1)

    def __init_cls_layers(self):
        cls_modules = []
        for _ in range(self.num_fc):
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.cls_modules = nn.ModuleList(cls_modules)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

    def __init_category_layers(self):
        if not self.enable_category:
            return
        category_modules = []
        for _ in range(self.num_fc):
            category_modules += [*LinearModule(self.fc_hidden_dim)]
        self.category_modules = nn.ModuleList(category_modules)
        self.prototypes = nn.Parameter(torch.randn(self.num_lane_categories, self.fc_hidden_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.01)

    def __init_attribute_layers(self):
        if not self.enable_attribute:
            return
        attribute_modules = []
        for _ in range(self.num_fc):
            attribute_modules += [*LinearModule(self.fc_hidden_dim)]
        self.attribute_modules = nn.ModuleList(attribute_modules)
        self.attribute_layers = nn.Linear(self.fc_hidden_dim, self.num_lr_attributes)

    def __init_buffers(self):
        self.register_buffer(
            name='sample_x_indexs',
            tensor=(torch.linspace(0, 1, steps=self.sample_points, dtype=torch.float32) * self.n_strips).long(),
        )
        self.register_buffer(
            name='prior_feat_ys',
            tensor=torch.flip((1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]),
        )
        self.register_buffer(
            name='sample_ys',
            tensor=torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32),
        )

    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)

    def _init_prior_embeddings_default(self):
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8
        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.0)
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.16 if i % 2 == 0 else 0.32)
        for i in range(left_priors_nums, left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 1],
                ((i - left_priors_nums) // 4 + 1) * bottom_strip_size,
            )
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.2 * (i % 4 + 1))
        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) * strip_size,
            )
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.0)
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.68 if i % 2 == 0 else 0.84)

    def _init_prior_embeddings(self):
        self._init_prior_embeddings_default()

    def generate_priors_from_embeddings(self):
        device = self.prior_embeddings.weight.device
        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        priors = torch.zeros((self.num_priors, 6 + self.n_offsets), device=device)
        priors[:, 2:5] = self.prior_embeddings.weight.clone()  # y, x, theta
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) * (self.img_w - 1)
            + (
                (
                    self.sample_ys.repeat(self.num_priors, 1).to(device)
                    - priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)
                )
                * self.img_h
                / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(1, self.n_offsets) * math.pi + 1e-5)
            )
        ) / (self.img_w - 1)
        priors_on_featmap = priors[..., 6 + self.sample_x_indexs]
        return priors, priors_on_featmap

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        batch_size = batch_features.shape[0]
        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        sample_ys = self.prior_feat_ys.view(1, 1, -1, 1).expand(batch_size, num_priors, -1, 1)
        prior_xs = prior_xs * 2.0 - 1.0
        grid_y = (sample_ys * 2.0) - 1.0
        grid = torch.cat((prior_xs, grid_y), dim=-1)
        feature = F.grid_sample(batch_features, grid, align_corners=True)
        feature = feature.permute(0, 2, 1, 3).reshape(
            batch_size * num_priors, self.prior_feat_channels, self.sample_points, 1
        )
        return feature

    def forward(self, features, img_metas=None, **kwargs):
        """
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: input features (list[Tensor])
        Return:
            dict:
                predictions_lists: list[lines], lines(B, num_priors, 6 + self.n_offsets) --- only when train
                last_pred_lanes: (B, num_priors, 6 + self.n_offsets)      --- only when eval
                seg: (B, num_priors, self.num_classes)               --- only when train
                category: (B, num_priors, self.num_lane_categories)  --- always, need to softmax
                attributes: (B, num_priors, self.num_lr_attributes)  --- always, need to softmax
        """
        batch_features = list(features[-self.refine_layers :])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        if self.training:
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        priors = self.priors.expand(batch_size, -1, -1)
        priors_on_featmap = self.priors_on_featmap.expand(batch_size, -1, -1)

        predictions_lists = []
        final_fc_features = None
        prior_features_stages = []

        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])
            batch_prior_features = self.pool_prior_features(batch_features[stage], num_priors, prior_xs)
            prior_features_stages.append(batch_prior_features)
            fc_features = self.roi_gather(prior_features_stages, batch_features[stage], stage)
            if stage == self.refine_layers - 1:
                final_fc_features = fc_features
            # Flatten (B, N, C) -> (B*N, C) without scrambling
            fc_features_flat = fc_features.view(batch_size * num_priors, self.fc_hidden_dim)

            cls_features = fc_features_flat.clone()
            reg_features = fc_features_flat.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])  # (B, num_priors, 2)
            reg = reg.reshape(batch_size, -1, reg.shape[1])

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits

            predictions[:, :, 2:5] += reg[:, :, :3]  # also reg theta angle here
            predictions[:, :, 5] = reg[:, :, 3]  # length

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1)
                + (
                    (1 - self.sample_ys.repeat(batch_size, num_priors, 1) - tran_tensor(predictions[..., 2]))
                    * self.img_h
                    / torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5)
                )
            ) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]

            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        seg_out = None
        category_out = None
        attribute_out = None
        outputs = {}

        if self.training:
            seg_features = torch.cat(
                [
                    F.interpolate(
                        feature,
                        size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                        mode='bilinear',
                        align_corners=False,
                    )
                    for feature in batch_features
                ],
                dim=1,
            )
            seg_out = self.seg_decoder(self.seg_conv(seg_features))
            outputs['predictions_lists'] = predictions_lists
            outputs.update(**seg_out)

        if self.enable_category and final_fc_features is not None:
            cat_feat = final_fc_features.clone()
            for m in self.category_modules:
                cat_feat = m(cat_feat)
            category_out = self.scale_factor * torch.matmul(
                F.normalize(cat_feat, p=2, dim=-1),
                F.normalize(self.prototypes, p=2, dim=-1).transpose(0, 1),
            )
            outputs['category'] = category_out.view(batch_size, self.num_priors, -1)

        if self.enable_attribute and final_fc_features is not None:
            attr_feat = final_fc_features.view(batch_size, self.num_priors, -1)
            for m in self.attribute_modules:
                attr_feat = m(attr_feat)
            attribute_out = self.attribute_layers(attr_feat)
            outputs['attribute'] = attribute_out

        if not self.training:
            final_preds = predictions_lists[-1]
            outputs['last_pred_lanes'] = final_preds
        return outputs

    def loss(self, outputs, batch, current_iter=0):
        device = self.priors.device
        predictions_lists = outputs['predictions_lists']  # [tensor(B,num_priors, 6 + num_offsets)]
        targets = batch['lane_line'].to(device)  # tensor(B, max_lanes, 6 + num_offsets)
        sample_xs = batch['sample_xs'].to(device)  # tensor(B, max_lanes, num_offsets)
        batch_lane_categories = batch.get('lane_categories', None).to(device)  # tensor(B， max_lanes)
        batch_lane_attributes = batch.get('lane_attributes', None).to(device)  # tensor(B， max_lanes)
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        targets_mask = targets[:, :, 1] == 1  # real lane gt mask

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_xytl_loss = torch.tensor(0.0, device=device)
        total_iou_loss = torch.tensor(0.0, device=device)
        total_category_loss = torch.tensor(0.0, device=device)
        total_attribute_loss = torch.tensor(0.0, device=device)
        log_ly = torch.tensor(0.0, device=device)
        log_lx = torch.tensor(0.0, device=device)
        log_lt = torch.tensor(0.0, device=device)
        log_ll = torch.tensor(0.0, device=device)

        total_stage_count = 0
        for stage in range(self.refine_layers):
            predictions = predictions_lists[stage]  # tensor(B, num_priors, 6 + num_offsets)
            with torch.no_grad():
                assigned_mask, assigned_ids = assign(
                    predictions,
                    targets,
                    targets_mask,
                    self.img_w,
                    self.img_h,
                    self.cfg,
                    current_iter=current_iter,
                    sample_ys=self.sample_ys,
                )

            cls_targets = assigned_mask.long().view(-1)
            pred_cls_flat = predictions[..., :2].view(-1, 2)
            num_positives = assigned_mask.sum()
            cls_norm = max(num_positives.item(), 1.0)
            stage_cls_loss = cls_criterion(pred_cls_flat, cls_targets).sum()
            total_cls_loss += stage_cls_loss / cls_norm  # accumulate cls loss

            if num_positives < 1:
                continue

            # regression loss
            batch_idx, prior_idx = torch.where(assigned_mask)
            target_idx = assigned_ids[batch_idx, prior_idx]  # (num_positives,)
            pos_preds = predictions[batch_idx, prior_idx]  # (num_positives, 6 + num_offsets)
            pos_targets = targets[batch_idx, target_idx]  # (num_positives, 6 + num_offsets)
            sample_xs_targets = sample_xs[batch_idx, target_idx]  # (num_positives, num_offsets)
            sample_xs_preds, _, _ = self.decoder(pos_preds, self.sample_ys)  # decode predicted lanes
            loss_y = F.smooth_l1_loss(pos_preds[:, 2], pos_targets[:, 2], reduction='none')
            loss_x = F.smooth_l1_loss(pos_preds[:, 3], pos_targets[:, 3], reduction='none')
            loss_theta = F.smooth_l1_loss(pos_preds[:, 4], pos_targets[:, 4], reduction='none')
            loss_len = F.smooth_l1_loss(pos_preds[:, 5], pos_targets[:, 5], reduction='none')
            reg_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
            loss_components = torch.stack([loss_y, loss_x, loss_theta, loss_len], dim=1)
            total_reg_xytl_loss += (loss_components * reg_weights).mean()

            if self.detailed_loss_logger is not None:
                log_ly += loss_y.mean()
                log_lx += loss_x.mean()
                log_lt += loss_theta.mean()
                log_ll += loss_len.mean()
                total_stage_count += 1

            ious = line_iou(
                sample_xs_preds.squeeze(),
                sample_xs_targets,
                self.img_w,
                length=15,
                aligned=True,
            )
            total_iou_loss += (1 - ious).mean()

            # only add category/attribute loss in the last stage
            if stage == self.refine_layers - 1:
                if self.enable_category:
                    cat_preds = outputs['category'][assigned_mask]
                    cat_targets = batch_lane_categories[batch_idx, target_idx]
                    total_category_loss += (
                        self.category_criterion(F.log_softmax(cat_preds, dim=-1), cat_targets) / cls_norm
                    )
                if self.enable_attribute:
                    attr_preds = outputs['attribute'][assigned_mask]
                    attr_targets = batch_lane_attributes[batch_idx, target_idx]
                    total_attribute_loss += (
                        self.attribute_criterion(F.log_softmax(attr_preds, dim=-1), attr_targets) / cls_norm
                    )

        seg_loss = self.criterion(
            F.log_softmax(outputs['seg'], dim=1),
            batch['seg'].long(),
        )

        if self.detailed_loss_logger is not None and total_stage_count > 0:
            detailed_loss_dict = {
                'loss_start_y': log_ly.item() / total_stage_count,
                'loss_start_x': log_lx.item() / total_stage_count,
                'loss_theta': log_lt.item() / total_stage_count,
                'loss_length': log_ll.item() / total_stage_count,
                'reg_xytl_loss': total_reg_xytl_loss.item() / self.refine_layers,
                'cls_loss': total_cls_loss.item() / self.refine_layers,
                'iou_loss': total_iou_loss.item() / self.refine_layers,
                'loss_category': total_category_loss.item(),
                'loss_attribute': total_attribute_loss.item(),
                'seg_loss': seg_loss.item(),
            }
            self.detailed_loss_logger.log(None, detailed_loss_dict)

        # Warmup weights
        alpha = current_iter / (self.warmup_epochs * self.epoch_per_iter)
        alpha = max(0.0, min(1.0, alpha))
        current_category_loss_weight = self.start_category_loss_weight + alpha * (
            self.category_loss_weight - self.start_category_loss_weight
        )
        current_attribute_loss_weight = self.start_attribute_loss_weight + alpha * (
            self.attribute_loss_weight - self.start_attribute_loss_weight
        )

        losses = {}
        losses['cls_loss'] = total_cls_loss / self.refine_layers * self.cfg.cls_loss_weight
        losses['reg_xytl_loss'] = total_reg_xytl_loss / self.refine_layers * self.cfg.xyt_loss_weight
        losses['iou_loss'] = total_iou_loss / self.refine_layers * self.cfg.iou_loss_weight
        losses['seg_loss'] = seg_loss * self.cfg.seg_loss_weight
        losses['loss_category'] = total_category_loss * current_category_loss_weight
        losses['loss_attribute'] = total_attribute_loss * current_attribute_loss_weight
        return losses

    def predictions_to_pred(self, predictions):
        """
        Convert predictions to internal Lane structure for evaluation.
        """
        sample_ys = self.sample_ys.to(predictions.device).double()
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))), self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(sample_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~(
                (((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool)
            )
            lane_xs[end + 1 :] = -2
            lane_xs[:start][mask] = -2
            lane_ys = sample_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) + self.cfg.cut_height) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(
                points=points.cpu().numpy(),
                metadata={'start_x': lane[3], 'start_y': lane[2], 'conf': lane[1]},
            )
            lanes.append(lane)
        return lanes

    def get_lanes(self, output, as_lanes=True):
        """
        Convert model output to lanes.
        """
        softmax = nn.Softmax(dim=1)
        all_pred_lanes = output.get('last_pred_lanes', None)
        if all_pred_lanes is None:
            return []
        decoded = []
        for predictions in all_pred_lanes:
            # filter out the conf lower than conf threshold
            threshold = self.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions.detach().clone()
            nms_predictions = torch.cat([nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[..., 5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores,
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes,
            )
            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded
