import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor, beta: float = 1.0) -> Tensor:
        beta_x = beta * inputs
        return inputs * beta_x.sigmoid()


class Conv2DExtractor(nn.Module):
    """
    Provides inteface of convolutional extractor.

    Note:
        Do not use this class directly, use one of the sub classes.
        Define the 'self.conv' class variable.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """
    supported_activations = {
        "hardtanh": nn.Hardtanh(0, 20, inplace=True),
        "relu": nn.ReLU(inplace=True),
        "elu": nn.ELU(inplace=True),
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "gelu": nn.GELU(),
        "swish": Swish(),
    }
    def __init__(self, d_input: int, activation: str = "hardtanh") -> None:
        super().__init__()
        self.d_input = d_input
        self.activation = Conv2DExtractor.supported_activations[activation]
        self.conv: List[nn.Module] = None

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            inputs (Tensor): torch.FloatTensor (batch, time, dimension)
            input_lengths (Tensor): torch.IntTensor (batch)

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        # tensor 변환 후 conv layer에 넣기
        # inputs : [bs, 1 ,dimension, time]
        outputs, output_lengths = self.conv(inputs.unsqueeze(1).transpose(2, 3), input_lengths) # output: [bs, 128, dimension / 4, time / 2]

        bs, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2) # output: [bs, time, channels, seq_lengths]
        outputs = outputs.view(bs, seq_lengths, channels * dimension) # [bs, seq_lengths, channels * time]
        return outputs, output_lengths
        

    def get_output_dim(self) -> int:
        if isinstance(self, VGGEXtractor):
            # d_input 2로 나눠지면 (d_input - 1) * 2^5
            # d_input 2로 안나눠지면 d_input * 2^5
            d_output = (self.d_input - 1) << 5 if self.d_input % 2 else self.d_input << 5

        else:
            raise ValueError(f"Unsupported Extractor : {self.extractor}")

        return d_output

    def get_output_length(self, seq_lengths: Tensor):
        assert self.con is not None, "self.conv should be defined"

        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                # 다음주에 실제로 mel 넣어서 확인
                numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
                seq_lengths = numerator.float() / float(module.stride[1])
                seq_lengths = seq_lengths.int() + 1

            elif isinstance(module, nn.MaxPool2d):
                seq_lengths >>= 1

        return seq_lengths.int()


class VGGEXtractor(Conv2DExtractor):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf

    Args:
        input_dim (int): Dimension of input vector
        in_channels (int): Number of channels in the input image
        out_channels (int or tuple): Number of channels produced by the convolution
        activation (str): Activation function

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """
    def __init__(
        self, 
        d_input: int, 
        in_channels: int = 1,
        out_channels: int or tuple = (64, 128),
        activation: str = "hardtanh"
        ) -> None:
        super().__init__(d_input, activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            # input : [bs, 1, dimension, time]
            nn.Conv2d(in_channels, out_channels[0], kernel_size=3, padding=1, bias=False),  # output : [bs, out_channels[0](64), dimension, time]
            nn.BatchNorm2d(num_features=out_channels[0]),
            self.activation,
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, padding=1, bias=False), # output : [bs, out_channels[0](64), dimension, time]
            nn.BatchNorm2d(num_features=out_channels[0]),
            self.activation,
            nn.MaxPool2d(2, stride=2),  # output : [bs, out_channels[0](64), dimension / 2, time / 2]
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, padding=1, bias=False), # output : [bs, out_channels[0](64), dimension, time] # output : [bs, out_channels[1](128), dimension / 2, time / 2]
            nn.BatchNorm2d(num_features=out_channels[1]),
            self.activation,
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, padding=1, bias=False), # output : [bs, out_channels[0](64), dimension, time] # output : [bs, out_channels[1](128), dimension / 2, time / 2]
            nn.BatchNorm2d(num_features=out_channels[0]),
            self.activation,
            nn.MaxPool2d(2, stride=2), # output: [bs, out_channels[1](128), dimension / 4, time / 2]
        )


    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        return super().forward(inputs, input_lengths)
    


