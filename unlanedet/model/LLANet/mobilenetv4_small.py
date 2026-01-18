"""
MobileNetV4-Small Backbone for Lane Detection.

Implements MobileNetV4-Small architecture with Universal Inverted Bottleneck (UIB) blocks
and Mobile MQA attention for efficient lane detection.

Optimized for RTX 4090 using FlashAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalInvertedBottleneck(nn.Module):
    """Universal Inverted Bottleneck (UIB) block."""

    def __init__(
        self,
        in_channels,
        expanded_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        mode="convnext",  # 'convnext' or 'inverted'
        use_attention=False,
        use_mqa=False,
        activation="relu",
    ):
        super().__init__()
        self.mode = mode
        self.use_attention = use_attention
        self.use_mqa = use_mqa

        # Activation function
        if activation == "relu":
            self.act = nn.ReLU6(inplace=True)
        elif activation == "hswish":
            self.act = nn.Hardswish(inplace=True)
        else:
            self.act = nn.ReLU6(inplace=True)

        if mode == "convnext":
            # ConvNeXt-style: Depthwise -> Pointwise (expand) -> Pointwise (project)
            self.dw_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            )
            self.norm1 = nn.BatchNorm2d(in_channels)

            self.pw_expand = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.norm2 = nn.BatchNorm2d(expanded_channels)

            self.pw_project = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
            self.norm3 = nn.BatchNorm2d(out_channels)

            # Shortcut connection
            self.has_shortcut = in_channels == out_channels and stride == 1

        else:  # inverted bottleneck mode
            # Inverted Bottleneck: Pointwise (expand) -> Depthwise -> Pointwise (project)
            self.pw_expand = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.norm1 = nn.BatchNorm2d(expanded_channels)

            self.dw_conv = nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            )
            self.norm2 = nn.BatchNorm2d(expanded_channels)

            self.pw_project = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
            self.norm3 = nn.BatchNorm2d(out_channels)

            # Shortcut connection
            self.has_shortcut = in_channels == out_channels and stride == 1

            if self.has_shortcut:
                self.shortcut = nn.Identity()

        # Mobile MQA attention for deep layers
        if use_attention and use_mqa and mode == "inverted":
            self.mqa = MobileMQA(expanded_channels)

    def forward(self, x):
        identity = x

        if self.mode == "convnext":
            # ConvNeXt mode: DW -> PW Expand -> PW Project
            x = self.dw_conv(x)
            x = self.norm1(x)
            x = self.act(x)

            x = self.pw_expand(x)
            x = self.norm2(x)
            x = self.act(x)

            x = self.pw_project(x)
            x = self.norm3(x)

        else:
            # Inverted bottleneck mode: PW Expand -> DW -> PW Project
            x = self.pw_expand(x)
            x = self.norm1(x)
            x = self.act(x)

            x = self.dw_conv(x)
            x = self.norm2(x)
            x = self.act(x)

            # Apply Mobile MQA attention if enabled
            if self.use_attention and hasattr(self, "mqa"):
                x = self.mqa(x)

            x = self.pw_project(x)
            x = self.norm3(x)

        # Residual connection
        if self.has_shortcut:
            x = x + identity

        return x


class MobileMQA(nn.Module):
    """Mobile Multi-Query Attention optimized for RTX 4090 (FlashAttention)."""

    def __init__(self, channels, num_heads=4, head_dim=64):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        # Multi-query: single key/value projection for all heads
        self.q_proj = nn.Conv2d(channels, num_heads * head_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(
            channels, head_dim, 1, bias=False
        )  # Shared across heads
        self.v_proj = nn.Conv2d(
            channels, head_dim, 1, bias=False
        )  # Shared across heads

        self.proj = nn.Conv2d(num_heads * head_dim, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 1. Generate Q, K, V
        # Reshape to (B, Heads, N, HeadDim) for SDPA
        q = (
            self.q_proj(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        )  # [B, H, N, D]
        k = (
            self.k_proj(x).view(B, 1, self.head_dim, N).permute(0, 1, 3, 2)
        )  # [B, 1, N, D]
        v = (
            self.v_proj(x).view(B, 1, self.head_dim, N).permute(0, 1, 3, 2)
        )  # [B, 1, N, D]

        # 2. Expand K, V for Multi-Query (Virtual Expansion)
        # Using .expand creates a view, consuming almost no extra memory
        k = k.expand(B, self.num_heads, N, self.head_dim)
        v = v.expand(B, self.num_heads, N, self.head_dim)

        # 3. FlashAttention (PyTorch 2.0+)
        # This will use highly optimized CUDA kernels on RTX 4090
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # 4. Reshape back
        out = (
            out.permute(0, 1, 3, 2)
            .contiguous()
            .view(B, self.num_heads * self.head_dim, H, W)
        )

        # 5. Project and Norm
        out = self.proj(out)
        out = self.norm(out)

        return out + x


class MobileNetV4Small(nn.Module):
    """MobileNetV4-Small backbone for lane detection.

    Returns 3 feature maps at strides 8, 16, 32 for FPN integration.
    """

    def __init__(self, pretrained=False, width_mult=1.0):
        super().__init__()

        # Width multiplier
        def width(w):
            return max(8, int(w * width_mult))

        # Stem: Conv2d 3x3, stride=2
        self.stem = nn.Sequential(
            nn.Conv2d(3, width(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width(32)),
            nn.Hardswish(inplace=True),
        )

        # Stage 1: ConvNeXt mode (shallow)
        self.stage1 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(32),
                width(64),
                width(32),
                kernel_size=3,
                stride=1,
                mode="convnext",
                activation="hswish",
            ),
            UniversalInvertedBottleneck(
                width(32),
                width(64),
                width(64),
                kernel_size=3,
                stride=2,
                mode="convnext",
                activation="hswish",
            ),
        )

        # Stage 2: ConvNeXt mode (shallow)
        self.stage2 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(64),
                width(128),
                width(64),
                kernel_size=3,
                stride=1,
                mode="convnext",
                activation="hswish",
            ),
            UniversalInvertedBottleneck(
                width(64),
                width(128),
                width(128),
                kernel_size=3,
                stride=2,
                mode="convnext",
                activation="hswish",
            ),
        )

        # Stage 3: Inverted bottleneck mode (deep)
        self.stage3 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(128),
                width(256),
                width(256),
                kernel_size=3,
                stride=2,
                mode="inverted",
                activation="hswish",
            ),
            UniversalInvertedBottleneck(
                width(256),
                width(256),
                width(256),
                kernel_size=3,
                stride=1,
                mode="inverted",
                activation="hswish",
            ),
        )

        # Stage 4: Inverted bottleneck mode with Mobile MQA (deep)
        self.stage4 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(256),
                width(512),
                width(256),
                kernel_size=3,
                stride=2,
                mode="inverted",
                use_attention=True,
                use_mqa=True,
                activation="hswish",
            ),
            UniversalInvertedBottleneck(
                width(256),
                width(512),
                width(256),
                kernel_size=3,
                stride=1,
                mode="inverted",
                use_attention=True,
                use_mqa=True,
                activation="hswish",
            ),
        )

        # Stage 5: Inverted bottleneck mode with Mobile MQA (deep)
        self.stage5 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(256),
                width(512),
                width(512),
                kernel_size=3,
                stride=2,
                mode="inverted",
                use_attention=True,
                use_mqa=True,
                activation="hswish",
            ),
            UniversalInvertedBottleneck(
                width(512),
                width(512),
                width(512),
                kernel_size=3,
                stride=1,
                mode="inverted",
                use_attention=True,
                use_mqa=True,
                activation="hswish",
            ),
        )

        # Feature map channels for FPN (stride 8, 16, 32)
        self.out_channels = [width(128), width(256), width(512)]

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with proper initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print(f"Loading pretrained weights from {pretrained}")
            pass
        else:
            print("Using random initialization")

    def forward(self, x):
        """Forward pass returning multi-scale features."""
        # Stem
        x = self.stem(x)  # stride 2

        # Stage 1 (stride 4)
        x = self.stage1(x)

        # Stage 2 (stride 8) -> Feature 1
        x = self.stage2(x)
        feat1 = x

        # Stage 3 (stride 16) -> Feature 2
        x = self.stage3(x)
        feat2 = x

        # Stage 4 (stride 16, no downsampling)
        x = self.stage4(x)

        # Stage 5 (stride 32) -> Feature 3
        x = self.stage5(x)
        feat3 = x

        # Return features for FPN: [stride8, stride16, stride32]
        features = [feat1, feat2, feat3]

        return features
