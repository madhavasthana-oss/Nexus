import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from enum import Enum

"""Refined modern neural-network building blocks with string support.

Key improvements applied in this fixed version:
- InstanceNorm2d created with affine=True (so it can learn scale/shift).
- AntiAliasedDownsample no longer relies on assigning Conv2d weights manually;
  it registers a fixed blur kernel buffer and uses F.conv2d (avoids device/param issues).
- ModulatedConv2d: added eps parameter for demodulation, contiguous safety,
  corrected bias broadcasting and weight initialization.
- Safety guards for GroupNorm and ECA kernel computation.
- Minor other safety and clarity fixes.
"""

# -------------------------------
# ENUMS & UTILITIES
# -------------------------------
class NormType(Enum):
    BATCH = "batch"
    INSTANCE = "instance"
    LAYER = "layer"
    GROUP = "group"
    NONE = "none"

class ActivationType(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    SWISH = "swish"
    MISH = "mish"
    GELU = "gelu"
    NONE = "none"

def parse_norm_type(norm_type: Union[str, NormType]) -> NormType:
    """Convert string to NormType enum."""
    if isinstance(norm_type, str):
        try:
            return NormType(norm_type.lower())
        except ValueError:
            valid_options = [e.value for e in NormType]
            raise ValueError(f"Invalid norm_type '{norm_type}'. Valid options: {valid_options}")
    return norm_type

def parse_activation_type(activation: Union[str, ActivationType]) -> ActivationType:
    """Convert string to ActivationType enum."""
    if isinstance(activation, str):
        try:
            return ActivationType(activation.lower())
        except ValueError:
            valid_options = [e.value for e in ActivationType]
            raise ValueError(f"Invalid activation '{activation}'. Valid options: {valid_options}")
    return activation

def get_norm_layer(norm_type: Union[str, NormType], channels: int, groups: int = 8) -> nn.Module:
    norm_type = parse_norm_type(norm_type)
    if norm_type == NormType.BATCH:
        return nn.BatchNorm2d(channels)
    if norm_type == NormType.INSTANCE:
        # allow learnable affine parameters for InstanceNorm
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type == NormType.LAYER:
        # LayerNorm compatible with conv features -> use GroupNorm with 1 group
        return nn.GroupNorm(1, channels)
    if norm_type == NormType.GROUP:
        # ensure groups <= channels and >0
        g = max(1, min(groups, channels))
        return nn.GroupNorm(g, channels)
    return nn.Identity()

def get_activation(act: Union[str, ActivationType], negative_slope: float = 0.2) -> nn.Module:
    act = parse_activation_type(act)
    if act == ActivationType.RELU:
        return nn.ReLU()
    if act == ActivationType.LEAKY_RELU:
        return nn.LeakyReLU(negative_slope)
    if act == ActivationType.ELU:
        return nn.ELU()
    if act == ActivationType.SELU:
        return nn.SELU()
    if act == ActivationType.SWISH:
        return nn.SiLU()  # SiLU is swish
    if act == ActivationType.MISH:
        return nn.Mish()
    if act == ActivationType.GELU:
        return nn.GELU()
    return nn.Identity()

def apply_spectral_norm(module: nn.Module, use_spectral: bool = False) -> nn.Module:
    return nn.utils.spectral_norm(module) if use_spectral and hasattr(module, 'weight') else module

# -------------------------------
# ATTENTION MODULES
# -------------------------------
# SEBlock: Squeeze-and-Excitation block for channel-wise attention.
# What it does: Enhances feature representation by recalibrating channel weights.
# How it works: Uses global average pooling to summarize spatial information, followed by two fully connected layers and sigmoid gating to produce channel-wise attention weights.
# Why and when to use: Useful in convolutional networks to emphasize important channels and suppress less relevant ones, improving performance in tasks like image classification. Use when channel-wise feature recalibration is beneficial.
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (channel-wise):

        Purpose: Improves representational power by adaptively recalibrating channel weights.
        How it works: Performs global average pooling → two FC layers → sigmoid gating to produce
        per-channel attention weights, which rescale input channels.
        Why useful: Lets the network emphasize informative features and suppress irrelevant ones.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ECABlock: Efficient Channel Attention block.
# What it does: Provides lightweight channel attention without dimensionality reduction.
# How it works: Applies global average pooling, followed by a 1D convolution with an adaptive kernel size to compute channel-wise attention weights, finalized with sigmoid gating.
# Why and when to use: Ideal for resource-constrained models where efficient channel attention is needed, such as in mobile or embedded vision applications. Use when parameter efficiency is critical.
class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA).

        Purpose: A lightweight alternative to SE that avoids dimensionality reduction.
        How it works: Computes global average pooling → applies a 1D convolution along channels
        with adaptive kernel size → sigmoid gating to produce per-channel weights.
        Why useful: Provides channel attention with lower cost and better parameter efficiency.
    """
    def __init__(self, channels: int, gamma: float = 2.0, b: int = 1):
        super().__init__()
        # compute adaptive kernel size similar to ECA paper
        # guard channels >= 1
        ch = max(1, channels)
        # compute t, ensure it's at least 1
        # use math.log2 for deterministic python scalar instead of torch ops here
        import math
        t = int(abs((math.log2(ch) + b) / gamma))
        k = t if (t % 2 == 1 and t >= 3) else max(3, t + 1 if t % 2 == 0 else 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # conv1d processes channel dimension: input shape -> (batch, 1, channels)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)          # (b, c)
        y = y.unsqueeze(1)                       # (b, 1, c)
        y = self.conv(y)                         # (b, 1, c)
        y = self.act(y).squeeze(1).view(b, c, 1, 1)
        return x * y

# CBAM: Convolutional Block Attention Module.
# What it does: Combines channel and spatial attention to focus on important features.
# How it works: First applies SE-like channel attention, then performs spatial attention by concatenating average and max pooling, followed by a convolution and sigmoid gating.
# Why and when to use: Enhances both channel and spatial feature focus, improving performance in tasks requiring precise localization, like object detection or segmentation. Use when both channel and spatial attention are beneficial.
class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

        Purpose: Sequentially applies channel attention and spatial attention.
        How it works: First applies SE-like channel gating → then spatial attention using avg+max pooling
        concatenated across channels → conv → sigmoid gating.
        Why useful: Helps the network focus on both important channels and spatial regions.
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        # spatial attention operates on concatenated avg/max along channel -> 2 channels
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * sa

# -------------------------------
# CORE: ConvBlock (with string support)
# -------------------------------
# ConvBlock: Basic convolutional block with normalization and activation.
# What it does: Serves as a fundamental feature extraction unit in neural networks.
# How it works: Applies a convolution, followed by normalization (batch, instance, layer, or group), an activation function (ReLU, LeakyReLU, etc.), and optional dropout.
# Why and when to use: Provides a configurable building block for feature extraction, ensuring stability through normalization and activation. Use in any convolutional architecture requiring standardized processing, such as CNNs for classification or segmentation.
class ConvBlock(nn.Module):
    """Basic Conv → Norm → Activation (+ optional Dropout).

        Purpose: Most common building block for feature extraction.
        How it works: Applies convolution, normalization (batch/instance/layer/group),
        chosen activation (ReLU, LeakyReLU, etc.), and optional dropout.
        Why useful: Provides configurable standardized feature extraction with stability.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_type: Union[str, NormType] = 'batch',
        activation: Union[str, ActivationType] = 'relu',
        dropout: float = 0.0,
        spectral_norm: bool = False,
        negative_slope:float = 0.2
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.conv = apply_spectral_norm(conv, spectral_norm)
        self.norm = get_norm_layer(norm_type, out_channels)
        self.activation = get_activation(activation,negative_slope = negative_slope)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

# -------------------------------
# Anti-aliased downsampling (robust device handling)
# -------------------------------
# AntiAliasedDownsample: Downsampling block with anti-aliasing.
# What it does: Reduces spatial resolution while minimizing aliasing artifacts.
# How it works: Applies a fixed blur kernel ([1,2,1]x[1,2,1]) before average pooling to smooth the input, ensuring better information preservation.
# Why and when to use: Prevents aliasing during downsampling, improving feature quality in tasks like image processing or generative models. Use in encoder or discriminator networks where downsampling is needed.
class AntiAliasedDownsample(nn.Module):
    """Anti-aliased downsampling block.

        Purpose: Prevents aliasing artifacts when reducing spatial resolution.
        How it works: Applies a fixed blur kernel ([1,2,1]x[1,2,1]) before AvgPool2d.
        Why useful: Produces smoother downsampling that preserves information better than naive pooling.
    """
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        if stride == 1:
            # no blur required
            self.register_buffer("_kernel", None)
            self.blur = False
        else:
            # create separable 3x3 kernel [1,2,1] x [1,2,1] normalized
            kernel = torch.tensor([1., 2., 1.])
            kernel = kernel[None, :] * kernel[:, None]
            kernel = kernel / kernel.sum()
            # register kernel as buffer so it moves with module to device
            # shape will be (1,1,3,3) and we will expand at forward time
            self.register_buffer("_kernel", kernel.view(1, 1, 3, 3))
            self.blur = True

        self.pool = nn.AvgPool2d(stride, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.blur:
            return self.pool(x)
        # expand kernel to match channels and use grouped conv via F.conv2d
        b, c, h, w = x.shape
        kernel = self._kernel.to(x.dtype)
        kernel = kernel.repeat(c, 1, 1, 1)  # (c,1,3,3)
        # Perform depthwise conv using groups=c
        x = F.conv2d(x, kernel, padding=1, groups=c)
        return self.pool(x)

# -------------------------------
# Residual & Bottleneck blocks (with string support)
# -------------------------------
# ResidualBlock: ResNet-style residual block with skip connections.
# What it does: Enhances gradient flow and feature representation by adding input to the output of two convolutional layers.
# How it works: Applies two convolutions with normalization and activation, optional attention, and a skip connection (with downsampling if needed).
# Why and when to use: Core component of ResNet architectures, preventing vanishing gradients in deep networks. Use in deep CNNs for tasks like image classification or object detection.
class ResidualBlock(nn.Module):
    """Residual block (ResNet-style).

        Purpose: Improves gradient flow and representation by using skip connections.
        How it works: Two conv layers with norm + activation, optional attention,
        added to skip connection (with optional downsampling).
        Why useful: Core building block of ResNets, prevents vanishing gradients.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm_type: Union[str, NormType] = 'batch',
        activation: Union[str, ActivationType] = 'relu',
        attention: Optional[str] = None,
        dropout: float = 0.0,
        anti_alias: bool = False,
        spectral_norm: bool = False,
        zero_init_residual: bool = False,
        negative_slope:float = 0.2
    ):
        super().__init__()
        mid_channels = out_channels

        # conv1: keep spatial size, then optionally downsample after conv1
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            dilation=dilation,
            groups=groups,
            bias=False,
            norm_type=norm_type,
            activation=activation,
            spectral_norm=spectral_norm,
            negative_slope=negative_slope
        )

        # conv2: produce out_channels
        self.conv2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            dilation=dilation,
            groups=groups,
            bias=False,
            norm_type=norm_type,
            activation='none',
            dropout=dropout,
            spectral_norm=spectral_norm,
            negative_slope=negative_slope
        )

        if attention == 'se':
            self.attention = SEBlock(out_channels)
        elif attention == 'eca':
            self.attention = ECABlock(out_channels)
        elif attention == 'cbam':
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()

        # Skip
        norm_type_enum = parse_norm_type(norm_type)
        if stride != 1 or in_channels != out_channels:
            skip_ops = []
            if anti_alias and stride > 1:
                # downsample on skip path then 1x1 conv
                skip_ops.append(AntiAliasedDownsample(in_channels, stride))
                skip_ops.append(apply_spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False), spectral_norm))
            else:
                skip_ops.append(apply_spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), spectral_norm))
            skip_ops.append(get_norm_layer(norm_type_enum, out_channels))
            self.skip = nn.Sequential(*skip_ops)
        else:
            self.skip = nn.Identity()

        # downsample applied to main path (after conv1) when stride > 1
        if stride > 1:
            self.downsample = AntiAliasedDownsample(out_channels if anti_alias else mid_channels, stride) if anti_alias else nn.AvgPool2d(stride, stride)
        else:
            self.downsample = nn.Identity()

        self.final_activation = get_activation(activation)

        if zero_init_residual and hasattr(self.conv2.norm, 'weight'):
            try:
                nn.init.constant_(self.conv2.norm.weight, 0)
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.downsample(out)
        out = self.conv2(out)
        out = self.attention(out)
        out = out + identity
        return self.final_activation(out)

# BottleneckBlock: Bottleneck residual block for efficient processing.
# What it does: Reduces computational cost by using a bottleneck structure in residual learning.
# How it works: Applies a 1x1 convolution to reduce channels, a 3x3 convolution for processing, and a 1x1 convolution to expand channels, with a skip connection.
# Why and when to use: Reduces parameters while maintaining representational power, as used in ResNet-50/101. Ideal for deep networks where efficiency is important, such as large-scale image classification.
class BottleneckBlock(nn.Module):
    """Bottleneck residual block.

        Purpose: Efficient residual block that reduces channels → processes → expands back.
        How it works: 1x1 reduce → 3x3 conv → 1x1 expand, plus skip connection.
        Why useful: Reduces parameters while retaining representational power (used in ResNet-50/101).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        groups: int = 1,
        norm_type: Union[str, NormType] = 'batch',
        activation: Union[str, ActivationType] = 'relu',
        attention: Optional[str] = None,
        dropout: float = 0.0,
        anti_alias: bool = False,
        spectral_norm: bool = False,
        negative_slope:float = 0.2
    ):
        super().__init__()
        mid = max(out_channels // expansion, 1)
        self.conv1 = ConvBlock(in_channels, mid, kernel_size=1, stride=1, norm_type=norm_type, activation=activation, spectral_norm=spectral_norm)
        self.conv2 = ConvBlock(mid, mid, kernel_size=3, stride=1, norm_type=norm_type, activation=activation, spectral_norm=spectral_norm)
        self.conv3 = ConvBlock(mid, out_channels, kernel_size=1, stride=1, norm_type=norm_type, activation='none', dropout=dropout, spectral_norm=spectral_norm)

        if attention == 'se':
            self.attention = SEBlock(out_channels)
        elif attention == 'eca':
            self.attention = ECABlock(out_channels)
        elif attention == 'cbam':
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()

        norm_type_enum = parse_norm_type(norm_type)
        if stride != 1 or in_channels != out_channels:
            if anti_alias and stride > 1:
                self.skip = nn.Sequential(
                    AntiAliasedDownsample(in_channels, stride),
                    apply_spectral_norm(nn.Conv2d(in_channels, out_channels, 1, bias=False), spectral_norm),
                    get_norm_layer(norm_type_enum, out_channels)
                )
            else:
                self.skip = nn.Sequential(
                    apply_spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), spectral_norm),
                    get_norm_layer(norm_type_enum, out_channels)
                )
        else:
            self.skip = nn.Identity()

        if stride > 1:
            self.downsample = AntiAliasedDownsample(mid, stride) if anti_alias else nn.AvgPool2d(stride, stride)
        else:
            self.downsample = nn.Identity()

        self.final_activation = get_activation(activation, negative_slope = negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        y = self.conv1(x)
        y = self.downsample(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.attention(y)
        y = y + identity
        return self.final_activation(y)

# -------------------------------
# Upsampling & style / generator oriented blocks
# -------------------------------
# UpsampleBlock: General-purpose upsampling block for increasing spatial resolution.
# What it does: Upsamples feature maps in generative or decoder networks.
# How it works: Supports multiple upsampling methods (ConvTranspose, bilinear+conv, or PixelShuffle), followed by normalization, activation, refinement convolution, and optional attention.
# Why and when to use: Flexible for high-quality upsampling in generative models like GANs or decoders in segmentation tasks. Use when upsampling is needed with customizable methods to reduce artifacts.
class UpsampleBlock(nn.Module):
    """General-purpose upsampling block.

        Purpose: Increases spatial resolution in generators/decoders.
        How it works: Can use ConvTranspose, bilinear+conv, or PixelShuffle for upsampling.
        Then refines features with conv/norm/activation and optional attention.
        Why useful: Provides flexible, artifact-free upsampling strategies.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        method: str = 'conv_transpose',
        norm_type: Union[str, NormType] = 'batch',
        activation: Union[str, ActivationType] = 'relu',
        attention: Optional[str] = None,
        dropout: float = 0.0,
        spectral_norm: bool = False,
        negative_slope:float = 0.2
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.method = method

        if method == 'conv_transpose':
            k = scale_factor + 1
            padding = k // 2
            output_padding = scale_factor - 1
            deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k,
                                        stride=scale_factor, padding=padding, output_padding=output_padding, bias=False)
            self.upsample = apply_spectral_norm(deconv, spectral_norm)
        elif method == 'interpolate':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                apply_spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), spectral_norm)
            )
        elif method == 'pixel_shuffle':
            conv = apply_spectral_norm(nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, 1, 1, bias=False), spectral_norm)
            self.upsample = nn.Sequential(conv, nn.PixelShuffle(scale_factor))
        else:
            raise ValueError(f"Unknown upsampling method: {method}")

        self.norm = get_norm_layer(norm_type, out_channels)
        self.activation = get_activation(activation,negative_slope = negative_slope)
        self.refine = ConvBlock(out_channels, out_channels, 3, norm_type=norm_type, activation=activation, dropout=dropout, spectral_norm=spectral_norm)

        if attention == 'se':
            self.attention = SEBlock(out_channels)
        elif attention == 'eca':
            self.attention = ECABlock(out_channels)
        elif attention == 'cbam':
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()

        norm_type_enum = parse_norm_type(norm_type)
        if in_channels != out_channels:
            if method == 'pixel_shuffle':
                self.skip = nn.Sequential(
                    apply_spectral_norm(nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 1, bias=False), spectral_norm),
                    nn.PixelShuffle(scale_factor),
                    apply_spectral_norm(nn.Conv2d(out_channels, out_channels, 1, bias=False), spectral_norm)
                )
            else:
                self.skip = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='nearest'), apply_spectral_norm(nn.Conv2d(in_channels, out_channels, 1, bias=False), spectral_norm))
        else:
            self.skip = nn.Upsample(scale_factor=scale_factor, mode='nearest')

        self.skip_norm = get_norm_layer(norm_type_enum, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.refine(out)
        out = self.attention(out)

        skip = self.skip(x)
        skip = self.skip_norm(skip)
        return out + skip

# -------------------------------
# Recurrent blocks (with string support)
# -------------------------------
# RecurrentBlock: Recurrent convolutional block with optional gating.
# What it does: Iteratively refines features through multiple convolutional passes.
# How it works: Applies a convolution, normalization, and activation repeatedly, with optional gating to control feature flow, adding residuals each step.
# Why and when to use: Enhances feature refinement in tasks requiring iterative processing, such as denoising or super-resolution. Use in networks needing iterative feature enhancement.
class RecurrentBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        steps: int = 2,
        norm_type: Union[str, NormType] = 'batch',
        activation: Union[str, ActivationType] = 'relu',
        gated: bool = True,
        spectral_norm: bool = False,
        negative_slope:float = 0.2
    ):
        super().__init__()
        self.steps = steps
        self.gated = gated
        self.conv = apply_spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False), spectral_norm)
        self.norm = get_norm_layer(norm_type, channels)
        self.activation = get_activation(activation, negative_slope = negative_slope)
        if gated:
            self.gate_conv = apply_spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=True), spectral_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            residual = x
            x = self.conv(x)
            x = self.norm(x)
            if self.gated:
                gate = torch.sigmoid(self.gate_conv(x))
                x = gate * self.activation(x) + (1 - gate) * residual
            else:
                x = self.activation(x) + residual
        return x

# -------------------------------
# Style-modulated conv (kept as-is with safety)
# -------------------------------
# ModulatedConv2d: Style-modulated convolutional layer.
# What it does: Modulates convolutional weights based on a style vector for adaptive feature generation.
# How it works: Scales convolution weights per-sample using a style vector, with optional demodulation to stabilize training, followed by a grouped convolution.
# Why and when to use: Essential for style-based generative models like StyleGAN, where per-sample style control is needed. Use in GANs or other generative tasks requiring style modulation.
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, style_dim: int, demodulate: bool = True, eps: float = 1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.eps = eps
        # weight is [out, in, k, k]
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.modulation = nn.Linear(style_dim, in_channels, bias=True)
        # stable init
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        if hasattr(self.modulation, 'bias') and self.modulation.bias is not None:
            nn.init.constant_(self.modulation.bias, 1.0)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x: (b, in_c, h, w)
        # style: (b, style_dim)
        b, in_c, h, w = x.shape
        assert in_c == self.in_channels, f"in_channels mismatch: got {in_c} vs {self.in_channels}"
        # modulation to scale per-sample
        style = self.modulation(style).view(b, 1, in_c, 1, 1).contiguous()  # (b,1,in,1,1)
        # broadcast weight -> (1, out, in, k, k)
        weight = self.weight.unsqueeze(0) * style  # (b, out, in, k, k)
        if self.demodulate:
            # demodulation per-sample per-out-channel
            demod = torch.rsqrt((weight.pow(2).sum([2, 3, 4]) + self.eps))  # (b, out)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        # reshape for grouped conv
        x = x.view(1, b * in_c, h, w)
        weight = weight.view(b * self.out_channels, in_c, self.kernel_size, self.kernel_size)
        out = F.conv2d(x, weight, padding=self.kernel_size // 2, groups=b)
        out = out.view(b, self.out_channels, h, w)
        # add bias properly
        out = out + self.bias.view(1, -1, 1, 1)
        return out

# -------------------------------
# Style-aware residual block (renamed to avoid collision)
# -------------------------------
# StyleResidualBlock: Compact residual block for style-aware architectures.
# What it does: Performs residual learning with batch normalization for simpler generator/discriminator setups.
# How it works: Uses two 3x3 convolutions with batch normalization and ReLU, with a skip connection to add the input to the output.
# Why and when to use: Simplifies residual learning for style-based networks with fewer parameters. Use in lightweight generators or discriminators, such as in basic GAN setups.
class StyleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=False)
        y = self.bn2(self.conv2(y))
        y = y + identity  # Replace += with +
        return F.relu(y, inplace=False)

# -------------------------------
# Residual upsampling blocks (with string support)
# -------------------------------
# ResidualUpBlock: Residual block with upsampling for generative tasks.
# What it does: Increases spatial resolution while maintaining residual learning for better feature propagation.
# How it works: Uses two transposed convolutions with batch normalization and ReLU, with a skip connection for residual learning.
# Why and when to use: Combines upsampling with residual connections for sharper outputs in generative models. Use in decoders or generators requiring upsampling, such as in image synthesis.
class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        output_padding = stride - 1 if stride > 1 else 0
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, stride, padding=1, output_padding=output_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, stride, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        y = F.relu(self.bn1(self.deconv1(x)),inplace = False)
        y = self.bn2(self.deconv2(y))
        y = y + identity
        return F.relu(y,inplace = False)

# BottleneckResidualUpBlock: Bottleneck-style residual upsampling block.
# What it does: Efficiently upsamples features using a bottleneck structure with residual learning.
# How it works: Reduces channels with a 1x1 transposed convolution, upsamples with a 3x3 transposed convolution, expands channels with a 1x1 convolution, and adds a skip connection.
# Why and when to use: Reduces computational cost while upsampling, suitable for deep generative models. Use in large-scale generative networks where efficiency and upsampling are needed, like in high-resolution image synthesis.
class BottleneckResidualUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, expansion: int = 4):
        super().__init__()
        assert stride >= 1
        mid = max(out_channels // expansion, 1)
        output_padding = stride - 1 if stride > 1 else 0
        self.reduce = nn.ConvTranspose2d(in_channels, mid, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.up = nn.ConvTranspose2d(mid, mid, 3, stride, padding=1, output_padding=output_padding, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.project = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, stride, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        y = F.relu(self.bn1(self.reduce(x)),inplace = False)
        y = F.relu(self.bn2(self.up(y)),inplace = False)
        y = self.bn3(self.project(y))
        y = y + identity
        return F.relu(y,inplace = False)

# -------------------------------
# Recurrent Upsampling Blocks
# -------------------------------
# RecurrentResidualUpBlock: Recurrent residual upsampling block.
# What it does: Upsamples features while iteratively refining them through recurrent convolutions.
# How it works: Performs upsampling with a transposed convolution, followed by iterative refinement using a convolutional layer, with a skip connection for residual learning.
# Why and when to use: Produces sharper high-resolution outputs by combining upsampling with iterative refinement. Use in generative tasks like super-resolution or image synthesis where high-quality upsampling is critical.
class RecurrentResidualUpBlock(nn.Module):
    """Recurrent residual upsampling block.

        Purpose: Performs upsampling while iteratively refining features.
        How it works: Upsamples via ConvTranspose, then applies multiple recurrent conv refinements
        before adding skip connection.
        Why useful: Produces sharper high-res outputs by combining residual learning and recurrence.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, t: int = 2):
        super().__init__()
        output_padding = stride - 1 if stride > 1 else 0
        self.t = t
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride, padding=1, output_padding=output_padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.refine_conv = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.refine_bn = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, stride, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        y = F.relu(self.bn(self.conv(x)),inplace = False)
        for _ in range(self.t - 1):
            y = F.relu(self.refine_bn(self.refine_conv(y)),inplace = False)
        y = y + identity
        return F.relu(y,inplace = False)
