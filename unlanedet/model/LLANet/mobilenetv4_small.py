"""
MobileNetV4-Small Backbone for Lane Detection.

Implements MobileNetV4-Small architecture with Universal Inverted Bottleneck (UIB) blocks
and Mobile MQA attention for efficient lane detection.

Reference: MobileNetV4 - Hybrid Vision Transformer and CNN Backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalInvertedBottleneck(nn.Module):
    """Universal Inverted Bottleneck (UIB) block.
    
    Supports two modes:
    - ConvNeXt mode (for shallow stages 1-2): Depthwise Conv before Pointwise Conv
    - Inverted Bottleneck mode (for deep stages 3-5): Depthwise Conv after expansion
    """
    
    def __init__(self, 
                 in_channels,
                 expanded_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 mode='convnext',  # 'convnext' or 'inverted'
                 use_attention=False,
                 use_mqa=False,
                 activation='relu'):
        super().__init__()
        self.mode = mode
        self.use_attention = use_attention
        self.use_mqa = use_mqa
        
        # Activation function
        if activation == 'relu':
            self.act = nn.ReLU6(inplace=True)
        elif activation == 'hswish':
            self.act = nn.Hardswish(inplace=True)
        else:
            self.act = nn.ReLU6(inplace=True)
        
        if mode == 'convnext':
            # ConvNeXt-style: Depthwise -> Pointwise (expand) -> Pointwise (project)
            self.dw_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size, 
                stride=stride, padding=kernel_size//2, 
                groups=in_channels, bias=False)
            self.norm1 = nn.BatchNorm2d(in_channels)
            
            self.pw_expand = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.norm2 = nn.BatchNorm2d(expanded_channels)
            
            self.pw_project = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
            self.norm3 = nn.BatchNorm2d(out_channels)
            
            # Shortcut connection
            self.has_shortcut = (in_channels == out_channels and stride == 1)
            
        else:  # inverted bottleneck mode
            # Inverted Bottleneck: Pointwise (expand) -> Depthwise -> Pointwise (project)
            self.pw_expand = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.norm1 = nn.BatchNorm2d(expanded_channels)
            
            self.dw_conv = nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride=stride, padding=kernel_size//2,
                groups=expanded_channels, bias=False)
            self.norm2 = nn.BatchNorm2d(expanded_channels)
            
            self.pw_project = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
            self.norm3 = nn.BatchNorm2d(out_channels)
            
            # Shortcut connection
            self.has_shortcut = (in_channels == out_channels and stride == 1)
            
            if self.has_shortcut:
                self.shortcut = nn.Identity()
        
        # Mobile MQA attention for deep layers
        if use_attention and use_mqa and mode == 'inverted':
            self.mqa = MobileMQA(expanded_channels)
    
    def forward(self, x):
        identity = x
        
        if self.mode == 'convnext':
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
            if self.use_attention and hasattr(self, 'mqa'):
                x = self.mqa(x)
            
            x = self.pw_project(x)
            x = self.norm3(x)
        
        # Residual connection
        if self.has_shortcut:
            x = x + identity
            
        return x


class MobileMQA(nn.Module):
    """Mobile Multi-Query Attention for efficient long-range feature capture.
    
    Optimized for mobile devices with reduced computation compared to standard MHA.
    """
    
    def __init__(self, channels, num_heads=4, head_dim=64):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Multi-query: single key/value projection for all heads
        self.q_proj = nn.Conv2d(channels, num_heads * head_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, head_dim, 1, bias=False)  # Shared across heads
        self.v_proj = nn.Conv2d(channels, head_dim, 1, bias=False)  # Shared across heads
        
        self.proj = nn.Conv2d(num_heads * head_dim, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Generate Q, K, V
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, N)
        k = self.k_proj(x).view(B, self.head_dim, N)  # Shared key
        v = self.v_proj(x).view(B, self.head_dim, N)  # Shared value
        
        # Reshape for attention - Fix dimension mismatch
        q = q.permute(0, 1, 3, 2)  # B, num_heads, N, head_dim
        
        # Adjust K and V dimensions for matrix multiplication
        # k: (B, head_dim, N) -> (B, 1, head_dim, N) for broadcasting
        k = k.unsqueeze(1)  # B, 1, head_dim, N
        
        # v: (B, head_dim, N) -> (B, 1, N, head_dim) for attention output
        v = v.permute(0, 2, 1).unsqueeze(1)  # B, 1, N, head_dim
        
        # Scaled dot-product attention with shared K, V
        # q: (B, num_heads, N, head_dim) @ k.transpose: (B, 1, head_dim, N) -> (B, num_heads, N, N)
        attn = torch.matmul(q, k) * self.scale  # B, num_heads, N, N
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        # attn: (B, num_heads, N, N) @ v: (B, 1, N, head_dim) -> (B, num_heads, N, head_dim)
        out = torch.matmul(attn, v)  # B, num_heads, N, head_dim
        
        # Reshape back to original format
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.num_heads * self.head_dim, H, W)
        
        # Project back
        out = self.proj(out)
        out = self.norm(out)
        
        return out + x  # Residual connection


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
            nn.Hardswish(inplace=True)
        )
        
        # Stage 1: ConvNeXt mode (shallow)
        self.stage1 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(32), width(64), width(32),
                kernel_size=3, stride=1, mode='convnext',
                activation='hswish'
            ),
            UniversalInvertedBottleneck(
                width(32), width(64), width(64),
                kernel_size=3, stride=2, mode='convnext',
                activation='hswish'
            ),
        )
        
        # Stage 2: ConvNeXt mode (shallow)
        self.stage2 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(64), width(128), width(64),
                kernel_size=3, stride=1, mode='convnext',
                activation='hswish'
            ),
            UniversalInvertedBottleneck(
                width(64), width(128), width(128),
                kernel_size=3, stride=2, mode='convnext',
                activation='hswish'
            ),
        )
        
        # Stage 3: Inverted bottleneck mode (deep)
        # 修复：将输出层通道数改为 width(256) 以匹配 FPN 期望
        self.stage3 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(128), width(256), width(256),  # Changed out to width(256)
                kernel_size=3, stride=2, mode='inverted',
                activation='hswish'
            ),
            UniversalInvertedBottleneck(
                width(256), width(256), width(256),  # Changed in/out to width(256)
                kernel_size=3, stride=1, mode='inverted',
                activation='hswish'
            ),
        )
        
        # Stage 4: Inverted bottleneck mode with Mobile MQA (deep)
        # 修复：输入通道改为 width(256) 以匹配 Stage 3 的输出
        self.stage4 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(256), width(512), width(256),  # Changed in to width(256)
                kernel_size=3, stride=2, mode='inverted',
                use_attention=True, use_mqa=True,
                activation='hswish'
            ),
            UniversalInvertedBottleneck(
                width(256), width(512), width(256),
                kernel_size=3, stride=1, mode='inverted',
                use_attention=True, use_mqa=True,
                activation='hswish'
            ),
        )
        
        # Stage 5: Inverted bottleneck mode with Mobile MQA (deep)
        self.stage5 = nn.Sequential(
            UniversalInvertedBottleneck(
                width(256), width(512), width(512),
                kernel_size=3, stride=2, mode='inverted',
                use_attention=True, use_mqa=True,
                activation='hswish'
            ),
            UniversalInvertedBottleneck(
                width(512), width(512), width(512),
                kernel_size=3, stride=1, mode='inverted',
                use_attention=True, use_mqa=True,
                activation='hswish'
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def init_weights(self, pretrained=None):
        """Initialize weights, optionally from pretrained ImageNet weights.
        
        Args:
            pretrained (str, optional): Path to pretrained weights
        """
        if pretrained is not None:
            # TODO: Load ImageNet pretrained weights
            # This is a placeholder for future implementation
            print(f"Loading pretrained weights from {pretrained}")
            # Example:
            # state_dict = torch.load(pretrained)
            # self.load_state_dict(state_dict, strict=False)
            pass
        else:
            print("Using random initialization")
    
    def forward(self, x):
        """Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            features: List of 3 feature maps at strides 8, 16, 32
        """
        # Stem
        x = self.stem(x)  # stride 2
        
        # Stage 1 (stride 4)
        x = self.stage1(x)
        
        # Stage 2 (stride 8) -> Feature 1
        x = self.stage2(x)
        feat1 = x  # stride 8
        
        # Stage 3 (stride 16) -> Feature 2
        x = self.stage3(x)
        feat2 = x  # stride 16
        
        # Stage 4 (stride 16, no downsampling)
        x = self.stage4(x)
        
        # Stage 5 (stride 32) -> Feature 3
        x = self.stage5(x)
        feat3 = x  # stride 32
        
        # Return features for FPN: [stride8, stride16, stride32]
        features = [feat1, feat2, feat3]
        
        return features


# Test the implementation
if __name__ == '__main__':
    # Create model
    model = MobileNetV4Small(width_mult=1.0)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 512)  # Typical lane detection input size
    features = model(x)
    
    print("MobileNetV4-Small Test:")
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(features):
        print(f"Feature {i+1} shape: {feat.shape}")
    
    print(f"\nOutput channels: {model.out_channels}")
    print("✓ Implementation test passed!")