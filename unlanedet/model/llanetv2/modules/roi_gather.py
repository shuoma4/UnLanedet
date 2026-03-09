import torch
import torch.nn as nn
import torch.nn.functional as F

from unlanedet.layers.batch_norm import get_norm
from unlanedet.layers.wrappers import Conv2d


class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, self.size)
        return x.flatten(2)


class ROIGather(nn.Module):
    """
    ROIGather module for gather global information
    """

    def __init__(
        self,
        in_channels,
        num_priors,
        sample_points,
        fc_hidden_dim,
        refine_layers,
        mid_channels=64,
        norm_type='BN',
    ):
        super(ROIGather, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        use_bias = norm_type == ''
        self.f_key = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
            norm=get_norm(norm=norm_type, out_channels=self.in_channels),
        )

        self.f_query = nn.Sequential(
            nn.Conv1d(
                in_channels=num_priors,
                out_channels=num_priors,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=num_priors,
            ),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W = nn.Conv1d(
            in_channels=num_priors,
            out_channels=num_priors,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=num_priors,
        )

        self.resize = FeatureResize()
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                Conv2d(
                    in_channels,
                    in_channels,
                    (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm=get_norm(norm=norm_type, out_channels=self.in_channels),
                )
            )

            self.catconv.append(
                Conv2d(
                    mid_channels * (i + 1),
                    in_channels,
                    (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm=get_norm(norm=norm_type, out_channels=self.in_channels),
                )
            )

        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        # x is a list of features from previous stages?
        # In original code: `prior_features_stages` list
        # x[i] is [Batch*num_priors, C, sample_points, 1]

        feats = []
        for i, feature in enumerate(x):
            # convs[i] applies to feature from stage i
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)

        # Concat along channel dim (dim=1)
        cat_feat = torch.cat(feats, dim=1)
        # Reduce channels
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        """
        Args:
            roi_features: list of prior features from previous stages
            x: global feature map
            layer_index: currently on which layer to refine
        Return:
            roi: prior features with gathered global information
        """
        # 1. ROI Features processing (local)
        roi = self.roi_fea(roi_features, layer_index)

        bs = x.size(0)
        # roi: [B*N, C, P, 1]
        roi = roi.contiguous().view(bs * self.num_priors, -1)

        # FC -> [B*N, C]
        roi = F.relu(self.fc_norm(self.fc(roi)))
        roi = roi.view(bs, self.num_priors, -1)

        # 2. Global Context Attention
        query = roi  # [B, N, C]

        value = self.resize(self.f_value(x))  # [B, C, H*W] -> [B, C, S] (S=250)
        query = self.f_query(query)  # [B, N, C] (conv1d on N groups)
        key = self.f_key(x)  # [B, C, H, W]
        key = self.resize(key)  # [B, C, S]

        value = value.permute(0, 2, 1)  # [B, S, C]

        # Attention: Q * K^T
        # Q: [B, N, C], K: [B, C, S] -> [B, N, S]
        sim_map = torch.matmul(query, key)
        sim_map = (self.in_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # Context: Attn * V
        # Sim: [B, N, S], V: [B, S, C] -> [B, N, C]
        context = torch.matmul(sim_map, value)

        # Transform context
        context = self.W(context)  # [B, N, C]

        # Add to ROI
        roi = roi + F.dropout(context, p=0.1, training=self.training)

        return roi


def LinearModule(hidden_dim):
    return nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
