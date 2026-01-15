import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Import necessary modules and classes
from ..module.head.plaindecoder import PlainDecoder
from ..module.losses.focal_loss import FocalLoss
from ..CLRNet.line_iou import line_iou, liou_loss
from ..CLRNet.dynamic_assign import assign
from ..CLRNet.roi_gather import ROIGather, LinearModule


class LLANetHead(nn.Module):
    """
    LLANet Head with Prototype-based Topological Head for Lane Attribute Prediction.

    This extends the original CLRHead to predict:
    - Lane existence (classification)
    - Lane geometry (regression) 
    - Lane category (15 categories for OpenLane: 13 lanes + 2 curbsides) - PROTOTYPE-BASED
    - Left-right attribute (4 categories for OpenLane)

    Key Innovation:
    - Category branch uses prototype-based cosine similarity instead of linear classifier
    - Maintains row-based anchor regression for lane geometry
    """

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
        enable_category=True,
        enable_attribute=True,
        num_lane_categories=15,  # OpenLane: 13 lanes + 2 curbsides
        scale_factor=20.0,  # Temperature scaling for cosine similarity
    ):
        super(LLANetHead, self).__init__()
        self.cfg = cfg
        self.enable_category = enable_category
        self.enable_attribute = enable_attribute
        self.num_lane_categories = num_lane_categories
        self.scale_factor = scale_factor  # Temperature coefficient for cosine similarity

        # Get configuration parameters
        if cfg is not None:
            self.img_w = self.cfg.img_w
            self.img_h = self.cfg.img_h
            self.num_lr_attributes = self.cfg.num_lr_attributes
        else:
            # Default values for testing
            self.img_w = 800
            self.img_h = 320
            self.num_lr_attributes = 4

        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        # Register buffers
        self.register_buffer(
            name="sample_x_indexs",
            tensor=(
                torch.linspace(0, 1, steps=self.sample_points, dtype=torch.float32)
                * self.n_strips
            ).long(),
        )
        self.register_buffer(
            name="prior_feat_ys",
            tensor=torch.flip(
                (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]
            ),
        )
        self.register_buffer(
            name="prior_ys",
            tensor=torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32),
        )

        self.prior_feat_channels = prior_feat_channels

        # Initialize prior embeddings
        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer(name="priors", tensor=init_priors)
        self.register_buffer(name="priors_on_featmap", tensor=priors_on_featmap)

        # Segmentation decoder
        # Segmentation features are concatenated from 3 refine layers (3 * 64 = 192 channels)
        # Add a 1x1 conv to reduce channels before PlainDecoder
        self.seg_conv = nn.Conv2d(
            self.prior_feat_channels * self.refine_layers, self.prior_feat_channels, 1
        )
        self.seg_decoder = PlainDecoder(cfg)

        # Regression and classification modules
        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        # ROI gathering
        self.roi_gather = ROIGather(
            self.prior_feat_channels,
            self.num_priors,
            self.sample_points,
            self.fc_hidden_dim,
            self.refine_layers,
        )

        # Output layers for detection
        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 + 1
        )  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

        # ===== PROTOTYPE-BASED CATEGORY PREDICTION =====
        if self.enable_category:
            # Feature transformation layers (keep the same as before for feature extraction)
            self.category_modules = nn.ModuleList(
                [
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            
            # REMOVED: self.category_layers = nn.Linear(self.fc_hidden_dim, self.num_lane_categories)
            
            # NEW: Learnable prototypes for each lane category
            # Shape: (num_lane_categories, fc_hidden_dim)
            self.prototypes = nn.Parameter(
                torch.randn(self.num_lane_categories, self.fc_hidden_dim)
            )
            
            # Initialize prototypes with small random values
            nn.init.normal_(self.prototypes, mean=0.0, std=0.01)

        # Attribute prediction branches (unchanged)
        if self.enable_attribute:
            self.attribute_modules = nn.ModuleList(
                [
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            self.attribute_layers = nn.Linear(
                self.fc_hidden_dim, self.num_lr_attributes
            )

        # Loss criterion
        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(
            ignore_index=self.cfg.ignore_label, weight=weights
        )

        # Attribute loss criteria
        if self.enable_category:
            # Use NLLLoss for prototype-based predictions (after log_softmax)
            self.category_criterion = torch.nn.NLLLoss()

        if self.enable_attribute:
            self.attribute_criterion = torch.nn.NLLLoss()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize layer weights."""
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        
        # Initialize prototype-based category layers
        if self.enable_category:
            for m in self.category_modules.parameters():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m, mean=0.0, std=1e-3)
            # Prototypes are initialized separately in __init__
        
        if self.enable_attribute:
            for m in self.attribute_layers.parameters():
                nn.init.normal_(m, mean=0.0, std=1e-3)

    def _init_prior_embeddings(self):
        """Initialize prior embeddings."""
        # Use nn.Embedding instead of separate parameters
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        nn.init.normal_(self.prior_embeddings.weight, mean=0.0, std=0.01)

    def generate_priors_from_embeddings(self):
        """Generate priors from learned embeddings."""
        device = self.prior_embeddings.weight.device

        # Generate priors in the format (num_priors, 6 + n_offsets)
        # 6 = 2 scores + start_x + start_y + theta + length
        # n_offsets = num_points
        priors = torch.zeros((self.num_priors, 6 + self.n_offsets), device=device)

        # Initialize with reasonable values
        priors[:, 0] = 0.1  # negative score
        priors[:, 1] = 0.9  # positive score
        priors[:, 2] = 0.5  # start_x (normalized)
        priors[:, 3] = 0.1  # start_y (normalized)
        priors[:, 4] = 0.0  # theta
        priors[:, 5] = 50.0  # length

        # Lane coordinates (linear spread across image)
        for i in range(self.num_priors):
            x_pos = (i + 0.5) / self.num_priors * (self.img_w - 1)
            for j in range(self.n_offsets):
                y_pos = (1 - j / (self.n_offsets - 1)) * (self.img_h - 1)
                priors[i, 6 + j] = x_pos / (self.img_w - 1)

        # Generate priors on feature map: extract sample_points positions
        priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        return priors, priors_on_featmap

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        """
        Pool prior feature from feature map.

        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W)
            num_priors: Number of priors
            prior_xs: Prior x coordinates on feature map, shape: (B, num_priors, sample_points, 1)

        Returns:
            feature: Pooled features, shape: (B*num_priors, C, sample_points, 1)
        """
        batch_size = batch_features.shape[0]
        device = batch_features.device

        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = (
            self.prior_feat_ys.to(device)
            .repeat(batch_size * num_priors)
            .view(batch_size, num_priors, -1, 1)
        )

        prior_xs = prior_xs * 2.0 - 1.0
        prior_ys = prior_ys * 2.0 - 1.0
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(batch_features, grid, align_corners=True).permute(
            0, 2, 1, 3
        )

        feature = feature.reshape(
            batch_size * num_priors, self.prior_feat_channels, self.sample_points, 1
        )
        return feature

    def forward(self, features, img_metas=None, **kwargs):
        """
        Forward pass.

        Args:
            features: Backbone+FPN output features (list of feature maps)
            img_metas: Image metadata (for compatibility with original CLRNet)
            **kwargs: Additional arguments like 'batch' from Detector

        Returns:
            Dictionary containing:
                - cls: Lane classification
                - reg: Lane regression
                - seg: Segmentation
                - category: Lane category (if enabled) - PROTOTYPE-BASED LOGITS
                - attribute: Lane left-right attribute (if enabled)
        """
        import math

        # Get batch features (use last refine_layers features)
        batch_features = list(features[-self.refine_layers :])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]
        device = batch_features[-1].device

        if self.training:
            # Generate priors on the correct device
            self.prior_embeddings = self.prior_embeddings.to(device)
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        # Ensure priors are on correct device
        self.priors = self.priors.to(device)
        self.priors_on_featmap = self.priors_on_featmap.to(device)

        priors, priors_on_featmap = self.priors.repeat(
            batch_size, 1, 1
        ), self.priors_on_featmap.to(device).repeat(batch_size, 1, 1)

        predictions_lists = []

        # Iterative refinement
        prior_features_stages = []
        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]  # Should be 192
            prior_xs = torch.flip(
                priors_on_featmap, dims=[2]
            )  # (B, 192, 36) → (B, 192, 36)

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs
            )
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(
                prior_features_stages, batch_features[stage], stage
            )

            # Fix dimension mismatch: roi_gather outputs (num_priors, batch, C) - Prior-First layout
            # Need to permute to Batch-First layout for downstream processing
            fc_features = fc_features.view(num_priors, batch_size, -1)
            # Permute to (batch_size, num_priors, C) and make contiguous
            fc_features = fc_features.permute(1, 0, 2).contiguous()
            # Flatten for FC layers: (B * num_priors, C)
            fc_features_flat = fc_features.view(batch_size * num_priors, self.fc_hidden_dim)

            cls_features = fc_features_flat.clone()
            reg_features = fc_features_flat.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1]
            )  # (B, num_priors, 2)
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
                    (
                        1
                        - self.prior_ys.to(device).repeat(batch_size, num_priors, 1)
                        - tran_tensor(predictions[..., 2])
                    )
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

        # Segmentation (only during training)
        if self.training:
            seg_features = torch.cat(
                [
                    F.interpolate(
                        feature,
                        size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for feature in batch_features
                ],
                dim=1,
            )
            # Reduce channels from 3 * 64 = 192 to 64 before decoder
            seg_features = self.seg_conv(seg_features)
            seg_out = self.seg_decoder(seg_features)
        else:
            seg_out = None

        # Attribute predictions (from final stage features)
        category_out = None
        attribute_out = None

        if self.training:
            # Use fc_features from final stage for attribute prediction
            if self.enable_category:
                # Extract category features through the same transformation layers
                # Use the unflattened fc_features (B, P, C) directly - no need to view
                category_features = fc_features.clone()  # fc_features is already (B, num_priors, C)
                for module in self.category_modules:
                    category_features = module(category_features)
                
                # ===== PROTOTYPE-BASED COSINE SIMILARITY CALCULATION =====
                # Normalize features and prototypes for cosine similarity
                normalized_features = F.normalize(category_features, p=2, dim=-1)  # [B, num_priors, hidden_dim]
                normalized_prototypes = F.normalize(self.prototypes, p=2, dim=-1)  # [num_categories, hidden_dim]
                
                # Compute cosine similarity matrix via dot product
                # normalized_features @ normalized_prototypes.T gives [B, num_priors, num_categories]
                cosine_sim = torch.matmul(normalized_features, normalized_prototypes.transpose(0, 1))
                
                # Apply temperature scaling
                category_out = self.scale_factor * cosine_sim  # [B, num_priors, num_categories]
                # Note: Return raw logits (before softmax) for CrossEntropy/Focal loss

            if self.enable_attribute:
                attribute_features = fc_features.view(batch_size, num_priors, -1)
                for module in self.attribute_modules:
                    attribute_features = module(attribute_features)
                attribute_out = self.attribute_layers(attribute_features)

            # Return outputs for training
            outputs = {"predictions_lists": predictions_lists}
            if seg_out is not None:
                outputs.update(**seg_out)
            if category_out is not None:
                outputs["category"] = category_out
            if attribute_out is not None:
                outputs["attribute"] = attribute_out

            return outputs
        else:
            # Return final predictions for inference
            return predictions_lists[-1]

    def loss(self, outputs, batch):
        """
        Compute losses based on CLRHead's loss method with additional attribute losses.
        Updated for prototype-based category prediction.

        Args:
            outputs: Model outputs
            batch: Ground truth batch

        Returns:
            Dictionary of losses
        """
        predictions_lists = outputs["predictions_lists"]
        targets = batch["lane_line"].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        cls_loss = torch.tensor(0.0).to(self.priors.device)
        reg_xytl_loss = torch.tensor(0.0).to(self.priors.device)
        iou_loss = torch.tensor(0.0).to(self.priors.device)

        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h
                    )

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= self.img_w - 1
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()

                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= self.img_w - 1
                reg_targets = target[matched_col_inds, 6:].clone()

                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        (predictions[matched_row_inds, 2] * self.n_strips)
                        .round()
                        .long(),
                        0,
                        self.n_strips,
                    )  # ensure the predictions starts is valid
                    target_starts = (
                        (target[matched_col_inds, 2] * self.n_strips).round().long()
                    )
                    target_yxtl[:, -1] -= (
                        predictions_starts - target_starts
                    )  # reg length

                # Loss calculation
                cls_loss = (
                    cls_loss
                    + cls_criterion(cls_pred, cls_target).sum() / target.shape[0]
                )

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180
                reg_xytl_loss = (
                    reg_xytl_loss
                    + F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction="none").mean()
                )

                iou_loss_value = liou_loss(
                    reg_pred, reg_targets, self.img_w, length=30
                )  # lane line length 30
                # Ensure iou_loss_value is a scalar
                if isinstance(iou_loss_value, torch.Tensor) and iou_loss_value.numel() > 1:
                    iou_loss_value = iou_loss_value.mean()
                iou_loss = iou_loss + iou_loss_value

        # extra segmentation loss
        # Skip seg_loss if all labels are 0 (no segmentation annotation)
        if batch["seg"].sum() > 0:
            seg_loss = self.criterion(
                F.log_softmax(outputs["seg"], dim=1), batch["seg"].long()
            )
            # Ensure seg_loss is a scalar
            if isinstance(seg_loss, torch.Tensor) and seg_loss.numel() > 1:
                seg_loss = seg_loss.mean()
        else:
            seg_loss = torch.tensor(0.0).to(self.priors.device)

        cls_loss /= len(targets) * self.refine_layers
        reg_xytl_loss /= len(targets) * self.refine_layers
        iou_loss /= len(targets) * self.refine_layers

        # Standard CLRNet losses
        losses = {}
        losses["cls_loss"] = cls_loss * self.cfg.cls_loss_weight
        losses["reg_xytl_loss"] = reg_xytl_loss * self.cfg.xyt_loss_weight
        losses["seg_loss"] = seg_loss * self.cfg.seg_loss_weight
        losses["iou_loss"] = iou_loss * self.cfg.iou_loss_weight

        # Attribute losses: only compute for matched priors (use final stage predictions)
        final_predictions = predictions_lists[-1]
        if self.enable_category or self.enable_attribute:
            for b in range(len(targets)):
                target = targets[b]
                # Check if target is valid and has positive samples
                if target.shape[0] == 0:
                    continue

                # Filter positive targets (positive score > 0.5)
                positive_mask = target[:, 1] > 0.5
                if positive_mask.sum() == 0:
                    continue

                target_positive = target[positive_mask]

                if len(target_positive) == 0:
                    continue

                with torch.no_grad():
                    predictions = final_predictions[b]
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target_positive, self.img_w, self.img_h
                    )

                if len(matched_row_inds) == 0:
                    continue

                # Attribute losses only for matched priors
                if self.enable_category and "category" in outputs:
                    category_preds = outputs["category"][b, matched_row_inds]  # Prototype-based logits
                    category_targets = batch["lane_categories"][b, matched_col_inds]

                    # Apply log_softmax to prototype similarities for NLLLoss
                    category_loss_b = self.category_criterion(
                        category_preds.log_softmax(dim=-1), category_targets
                    )
                    # Ensure category_loss_b is a scalar
                    if isinstance(category_loss_b, torch.Tensor) and category_loss_b.numel() > 1:
                        category_loss_b = category_loss_b.mean()

                    if "loss_category" not in losses:
                        losses["loss_category"] = torch.tensor(0.0).to(
                            self.priors.device
                        )
                    losses["loss_category"] = losses["loss_category"] + category_loss_b

                if self.enable_attribute and "attribute" in outputs:
                    attribute_preds = outputs["attribute"][b, matched_row_inds]
                    attribute_targets = batch["lane_attributes"][b, matched_col_inds]
                    attribute_loss_b = self.attribute_criterion(
                        attribute_preds.log_softmax(dim=-1), attribute_targets
                    )
                    # Ensure attribute_loss_b is a scalar
                    if isinstance(attribute_loss_b, torch.Tensor) and attribute_loss_b.numel() > 1:
                        attribute_loss_b = attribute_loss_b.mean()

                    if "loss_attribute" not in losses:
                        losses["loss_attribute"] = torch.tensor(0.0).to(
                            self.priors.device
                        )
                    losses["loss_attribute"] = (
                        losses["loss_attribute"] + attribute_loss_b
                    )

            # Average attribute losses across batch
            if "loss_category" in losses:
                losses["loss_category"] = losses["loss_category"] / len(targets)
            if "loss_attribute" in losses:
                losses["loss_attribute"] = losses["loss_attribute"] / len(targets)

            if self.enable_category and "loss_category" in losses:
                losses["loss_category"] = (
                    losses["loss_category"] * self.cfg.category_loss_weight
                )
            if self.enable_attribute and "loss_attribute" in losses:
                losses["loss_attribute"] = (
                    losses["loss_attribute"] * self.cfg.attribute_loss_weight
                )

        return losses