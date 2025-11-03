import torch
import torch.nn as nn

# Brevitas imports
from brevitas.nn import QuantConv2d, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

try:
    from source.models.RepConv_block import RepBlock, RepBlockV2
except ModuleNotFoundError:
    from RepConv_block import RepBlock, RepBlockV2


class QuantConv3X3(nn.Module):
    """
    Quantized 3x3 convolution block with configurable activation type.
    Compatible with Brevitas >= 0.12.0.
    """
    def __init__(self, inp_planes, out_planes, act_type='prelu', weight_bit_width=8, act_bit_width=8):
        super().__init__()

        

        # Define a weight quantizer subclass to override bit-width dynamically
        class CustomWeightQuant(Int8WeightPerTensorFloat):
            bit_width = weight_bit_width

        # Similarly, define activation quantizer subclass
        class CustomActQuant(Uint8ActPerTensorFloat):
            bit_width = act_bit_width
            
        self.quant_in = QuantIdentity(act_quant=CustomActQuant,return_quant_tensor=True)

        # Quantized convolution
        self.conv = QuantConv2d(
            in_channels=inp_planes,
            out_channels=out_planes,
            kernel_size=3,
            padding=1,
            bias=True,
            weight_quant=CustomWeightQuant,
            return_quant_tensor=True   # keep quant flow
        )

        # Activation
        if act_type == 'relu':
            self.act = QuantReLU(act_quant=CustomActQuant,return_quant_tensor=True)
        elif act_type == 'prelu':
            # Brevitas does not quantize PReLU trainable params directly; fallback to nn.PReLU
            self.act = nn.PReLU(num_parameters=out_planes)
        else:
            # Default to quantized ReLU if unspecified
            self.act = QuantReLU(act_quant=CustomActQuant,return_quant_tensor=True)

    def forward(self, x):
        x = self.quant_in(x) 
        y = self.conv(x)
        y = self.act(y)
        return y


class QuantPlainRepConv(nn.Module):
    """
    Quantized PlainRepConv model using Brevitas quantized layers.
    Compatible with Brevitas >= 0.12.0.
    """
    def __init__(self, module_nums=6, channel_nums=32, act_type='relu', scale=2, colors=3, bit_width=8):
        super(QuantPlainRepConv, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.bit_width = bit_width

        # Head
        self.head = QuantConv3X3(
            inp_planes=self.colors,
            out_planes=self.channel_nums,
            act_type=self.act_type,
            weight_bit_width=bit_width,
            act_bit_width=bit_width
        )

        # Backbone: sequential quantized conv pairs
        backbone = []
        for i in range(self.module_nums):
            backbone.append(
                nn.Sequential(
                    QuantConv3X3(
                        inp_planes=self.channel_nums,
                        out_planes=self.channel_nums,
                        act_type=self.act_type,
                        weight_bit_width=bit_width,
                        act_bit_width=bit_width
                    ),
                    QuantConv3X3(
                        inp_planes=self.channel_nums,
                        out_planes=self.channel_nums,
                        act_type=self.act_type,
                        weight_bit_width=bit_width,
                        act_bit_width=bit_width
                    )
                )
            )
        self.backbone = nn.Sequential(*backbone)

        # Transition
        self.transition = QuantConv3X3(
            inp_planes=self.channel_nums,
            out_planes=self.colors * (self.scale ** 2),
            act_type='linear',
            weight_bit_width=bit_width,
            act_bit_width=bit_width
        )

        # Upsampler
        self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x):
        y0 = self.head(x)
        y = y0
        for blk in self.backbone:
            y = blk(y)
            
         # ✅ Convert QuantTensors to regular tensors before addition
        if hasattr(y, 'tensor'):
            y = y.tensor
        if hasattr(y0, 'tensor'):
            y0 = y0.tensor
        
        y = self.transition(y + y0)
        y = torch.clamp(y, 0., 255.)
        y = self.upsampler(y)
        return y

    def fuse_model(self):
        # Not required for Brevitas quantization; placeholder for API consistency
        return

