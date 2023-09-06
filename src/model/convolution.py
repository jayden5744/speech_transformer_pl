import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor, beta: float = 1.0) -> Tensor:
        beta_x = beta * inputs
        return inputs * beta_x.sigmoid()


class MaskCNN(nn.Module):
    """
    Masking Convolutional Neural Network

    Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch sizes change during inference.
    Input needs to be in the shape of (batch_size, channel, hidden_dim, seq_len)

    Refer to https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Copyright (c) 2017 Sean Naren
    MIT License

    Args:
        sequential (torch.nn): sequential list of convolution layer

    Inputs: inputs, seq_lengths
        - **inputs** (torch.FloatTensor): The input of size BxCxHxT
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch

    Returns: output, seq_lengths
        - **output**: Masked output from the sequential
        - **seq_lengths**: Sequence length of output from the sequential
    """

    def __init__(self, sequential: nn.Sequential) -> None:
        super().__init__()
        self.sequential = sequential

    def foward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)

            mask = torch.BoolTensor(output.size()).fill_(0)

            seq_lengths = self._get_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                length = length.item()
                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(
                        dim=2, start=length, length=mask[idx].size(2) - length
                    ).fill_(1)
            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        """
        Calculate convolutional neural network receptive formula

        Args:
            module (torch.nn.Module): module of CNN
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch

        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """
        if isinstance(module, nn.Conv2d):
            numerator = (
                seq_lengths + 2
                & module.padding[1]
                - module.dilation[1] * (module.kernel_size[1] - 1)
                - 1
            )
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1  # / 2

        return seq_lengths.int()


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
        outputs, output_lengths = self.conv(
            inputs.unsqueeze(1).transpose(2, 3), input_lengths
        )  # output: [bs, 128, dimension / 4, time / 2]

        bs, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(
            0, 3, 1, 2
        )  # output: [bs, time, channels, seq_lengths]
        outputs = outputs.view(
            bs, seq_lengths, channels * dimension
        )  # [bs, seq_lengths, channels * time]
        return outputs, output_lengths

    def get_output_dim(self) -> int:
        if isinstance(self, VGGEXtractor):
            # d_input 2로 나눠지면 (d_input - 1) * 2^5
            # d_input 2로 안나눠지면 d_input * 2^5
            d_output = (
                (self.d_input - 1) << 5 if self.d_input % 2 else self.d_input << 5
            )

        else:
            raise ValueError(f"Unsupported Extractor : {self.extractor}")

        return d_output

    def get_output_length(self, seq_lengths: Tensor):
        assert self.con is not None, "self.conv should be defined"

        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                # 다음주에 실제로 mel 넣어서 확인
                numerator = (
                    seq_lengths
                    + 2 * module.padding[1]
                    - module.dilation[1] * (module.kernel_size[1] - 1)
                    - 1
                )
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
        activation: str = "hardtanh",
    ) -> None:
        super().__init__(d_input, activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskCNN(
            nn.Sequential(
                # input : [bs, 1, dimension, time]
                nn.Conv2d(
                    in_channels, out_channels[0], kernel_size=3, padding=1, bias=False
                ),  # output : [bs, out_channels[0](64), dimension, time]
                nn.BatchNorm2d(num_features=out_channels[0]),
                self.activation,
                nn.Conv2d(
                    out_channels[0],
                    out_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),  # output : [bs, out_channels[0](64), dimension, time]
                nn.BatchNorm2d(num_features=out_channels[0]),
                self.activation,
                nn.MaxPool2d(
                    2, stride=2
                ),  # output : [bs, out_channels[0](64), dimension / 2, time / 2]
                nn.Conv2d(
                    out_channels[0],
                    out_channels[1],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),  # output : [bs, out_channels[1](128), dimension / 2, time / 2]
                nn.BatchNorm2d(num_features=out_channels[1]),
                self.activation,
                nn.Conv2d(
                    out_channels[1],
                    out_channels[1],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),  # output : [bs, out_channels[1](128), dimension / 2, time / 2]
                nn.BatchNorm2d(num_features=out_channels[0]),
                self.activation,
                nn.MaxPool2d(
                    2, stride=2
                ),  # output: [bs, out_channels[1](128), dimension / 4, time / 4]
            )
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        return super().forward(inputs, input_lengths)
