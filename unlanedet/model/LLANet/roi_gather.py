from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers import Conv2d, get_norm


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
    Args:
        in_channels: prior feature channels
        num_priors: prior numbers we predefined
        sample_points: the number of sampled points when we extract feature from line
        fc_hidden_dim: the fc output channel
        refine_layers: the total number of layers to build refine
    """

    def __init__(
        self,
        in_channels,
        num_priors,
        sample_points,
        fc_hidden_dim,
        refine_layers,
        mid_channels=64,  # 给一个默认值，但通常会被覆盖
        norm_type="BN",
    ):
        super(ROIGather, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        use_bias = norm_type == ""
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
            ),

        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)

        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = torch.cat(feats, dim=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        """
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            x: feature map
            layer_index: currently on which layer to refine
        Return:
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        """
        roi = self.roi_fea(roi_features, layer_index)
        if torch.isnan(roi).any():
            print(f"NaN in roi after roi_fea at layer {layer_index}", flush=True)

        bs = x.size(0)
        roi = roi.contiguous().view(bs * self.num_priors, -1)

        roi_fc = self.fc(roi)
        if torch.isnan(roi_fc).any():
            print(
                f"NaN in roi after FC (before norm) at layer {layer_index}", flush=True
            )
        if torch.isinf(roi_fc).any():
            print(
                f"Inf in roi after FC (before norm) at layer {layer_index}", flush=True
            )

        roi = F.relu(self.fc_norm(roi_fc))
        if torch.isnan(roi).any():
            print(f"NaN in roi after fc_norm at layer {layer_index}", flush=True)

        roi = roi.view(bs, self.num_priors, -1)
        query = roi

        value = self.resize(self.f_value(x))
        if torch.isnan(value).any():
            print(f"NaN in value at layer {layer_index}", flush=True)

        query = self.f_query(query)
        if torch.isnan(query).any():
            print(f"NaN in query at layer {layer_index}", flush=True)

        key = self.f_key(x)
        if torch.isnan(key).any():
            print(f"NaN in key at layer {layer_index}", flush=True)

        value = value.permute(0, 2, 1)
        key = self.resize(key)
        sim_map = torch.matmul(query, key)
        if torch.isnan(sim_map).any():
            print(f"NaN in sim_map before softmax at layer {layer_index}", flush=True)

        sim_map = (self.in_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        if torch.isnan(sim_map).any():
            print(f"NaN in sim_map after softmax at layer {layer_index}", flush=True)

        context = torch.matmul(sim_map, value)
        if torch.isnan(context).any():
            print(f"NaN in context at layer {layer_index}", flush=True)

        context = self.W(context)

        roi = roi + F.dropout(context, p=0.1, training=self.training)
        if torch.isnan(roi).any():
            print(f"NaN in roi output at layer {layer_index}", flush=True)

        return roi


def LinearModule(hidden_dim):
    return [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
