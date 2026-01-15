import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Mock imports to avoid relative import issues
class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        norm=None,
        activation=None,
        **kwargs
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, **kwargs
        )


class BatchNorm2d(nn.BatchNorm2d):
    pass


def get_norm(norm_cfg, num_features):
    return BatchNorm2d(num_features)


class Activation:
    def __init__(self, act=None):
        self.act = nn.ReLU(inplace=True) if act is None else nn.ReLU(inplace=True)


class AsymmetricConv(nn.Module):
    """非对称卷积模块 - 并行使用 k×1 和 1×k 卷积"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, norm_cfg=None, act_cfg=None
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        # 允许投影到 FPN 目标维度（例如 64）
        # 移除强制 actual_out_channels = in_channels

        # 水平卷积 (k × 1)
        self.h_conv = Conv2d(
            in_channels,
            out_channels,  # 使用传入的 out_channels
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            norm=get_norm(norm_cfg, out_channels) if norm_cfg else None,
            activation=Activation(act=act_cfg) if act_cfg else None,
        )

        # 垂直卷积 (1 × k)
        self.v_conv = Conv2d(
            in_channels,
            out_channels,  # 使用传入的 out_channels
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            norm=get_norm(norm_cfg, out_channels) if norm_cfg else None,
            activation=Activation(act=act_cfg) if act_cfg else None,
        )

    def forward(self, x):
        h_out = self.h_conv(x)
        v_out = self.v_conv(x)
        return h_out + v_out


class SCModule(nn.Module):
    """Strip Context Module - 条带上下文模块"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, norm_cfg=None, act_cfg=None
    ):
        super().__init__()

        # 允许通道变换 (in_channels -> out_channels)
        self.asym_conv = AsymmetricConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # 注意力权重生成 (att_weight 通道数需与 context_feat 一致)
        self.att_conv = Conv2d(
            out_channels, out_channels, kernel_size=1, activation=nn.Sigmoid()
        )

    def forward(self, x):
        # 非对称卷积提取条带上下文特征并投影维度
        # x: (B, in_channels, H, W) -> context_feat: (B, out_channels, H, W)
        context_feat = self.asym_conv(x)

        # 生成空间注意力权重
        att_weight = self.att_conv(context_feat)

        # 应用注意力权重到变换后的特征上
        attended_feat = context_feat * att_weight
        return attended_feat


class GSAFPN(nn.Module):
    """
    Geometry-Semantic Alignment FPN (GSA-FPN)

    几何语义对齐特征金字塔网络，主要特点：
    1. SCM模块：在横向连接中使用非对称卷积和注意力机制
    2. 全局语义注入：将最深层的全局语义信息注入到浅层特征
    3. 保持多尺度特征输出的同时增强几何和语义对齐能力
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        extra_convs_on_inputs=True,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        attention=False,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
        scm_kernel_size=3,
        enable_global_semantic=True,
    ):
        super(GSAFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.attention = attention
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.enable_global_semantic = enable_global_semantic

        # 设置backbone结束层级
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                self.add_extra_convs = "on_input"
            else:
                self.add_extra_convs = "on_output"

        # ===== SCM模块 (Strip Context Module) =====
        # 替换标准1x1卷积为SCM模块
        self.scm_modules = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            scm = SCModule(
                in_channels=in_channels[i],
                out_channels=out_channels,  # 使用 FPN 统一输出通道数 (64)
                kernel_size=scm_kernel_size,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.scm_modules.append(scm)

        # FPN输出卷积
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm=get_norm(norm_cfg, out_channels),
                activation=Activation(act=act_cfg),
            )
            self.fpn_convs.append(fpn_conv)

        # ===== 全局语义注入 =====
        if self.enable_global_semantic and self.num_ins >= 3:
            # 全局平均池化层 - 用于提取c5的全局语义
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            # 语义投影层 - 将全局语义映射到特征维度
            self.semantic_projection = nn.Sequential(
                Conv2d(out_channels, out_channels, 1),
                nn.ReLU(inplace=True),
                Conv2d(out_channels, out_channels, 1),
            )

        # ===== 额外的卷积层（如果需要更多输出层级）=====
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            self.extra_fpn_convs = nn.ModuleList()
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels_extra = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels_extra = out_channels

                extra_fpn_conv = Conv2d(
                    in_channels_extra,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm=get_norm(norm_cfg, out_channels),
                    activation=Activation(act=act_cfg),
                )
                self.extra_fpn_convs.append(extra_fpn_conv)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """前向传播

        Args:
            inputs: List[Tensor], 骨干网络的多尺度特征图 [c3, c4, c5]
                    c3: stride=8,  c4: stride=16, c5: stride=32

        Returns:
            Tuple[Tensor]: 融合后的特征金字塔 [p3, p4, p5, ...]
        """
        assert len(inputs) >= len(self.in_channels)

        # 如果输入多于预期，删除前面的特征图（通常是更低层级的特征）
        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # ===== 1. SCM横向连接 =====
        laterals = [
            self.scm_modules[i](inputs[i + self.start_level])
            for i, _ in enumerate(self.scm_modules)
        ]

        # ===== 2. 全局语义注入 =====
        if self.enable_global_semantic and len(laterals) >= 3:
            # 对最深层特征(c5对应的lateral)进行全局平均池化
            deepest_feature = laterals[-1]  # c5对应的特征图
            global_semantic = self.global_avg_pool(deepest_feature)  # [B, C, 1, 1]

            # 将全局语义投影到特征空间
            semantic_feat = self.semantic_projection(global_semantic)  # [B, C, 1, 1]

            # 广播到浅层特征的尺寸并相加（语义对齐）
            # 注入到p4 (c4对应的特征图)
            if len(laterals) >= 2:
                semantic_p4 = F.interpolate(
                    semantic_feat,
                    size=laterals[-2].shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                laterals[-2] = laterals[-2] + semantic_p4

            # 注入到p3 (c3对应的特征图) - 增强鲁棒性检查
            if len(laterals) >= 1:
                target_idx = 0  # First lateral corresponds to p3/stride8
                semantic_p3 = F.interpolate(
                    semantic_feat,
                    size=laterals[target_idx].shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                laterals[target_idx] = laterals[target_idx] + semantic_p3

        # ===== 3. 构建自上而下的路径 =====
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 上采样高层特征并与低层特征融合
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )

        # ===== 4. 构建输出特征图 =====
        # 第一部分：原始层级的输出
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 第二部分：添加额外层级
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                # 使用最大池化获取更多层级
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # 在原始特征图上添加卷积层
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(self.extra_fpn_convs[0](extra_source))

                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(
                            self.extra_fpn_convs[i - used_backbone_levels](
                                F.relu(outs[-1])
                            )
                        )
                    else:
                        outs.append(
                            self.extra_fpn_convs[i - used_backbone_levels](outs[-1])
                        )

        return tuple(outs)
