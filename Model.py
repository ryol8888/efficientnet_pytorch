import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import utils





GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'se_coefficient', 'local_pooling', 'condconv_num_experts',
    'clip_projection_output', 'blocks_args', 'fix_head_stem',
])
# Note: the default value of None is not necessarily valid. It is valid to leave
# width_coefficient, depth_coefficient at None, which is treated as 1.0 (and
# which also allows depth_divisor and min_depth to be left at None).

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def swish(x): 
  return x * torch.sigmoid(x)

class EfficientNet(nn.Module):
  def __init__(self,blocks_args=None, global_params=None):
    """Initializes an `Model` instance.
    Args:
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.
    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super(EfficientNet, self).__init__()
      assert isinstance(blocks_args, list) 'blocks_args should be a list.'
    self._global_params = global_params
    self._blocks_args = blocks_args
    self._relu_fn = global_params.relu_fn or swish
    self._batch_norm = global_params.batch_norm
    self._fix_head_stem = global_params.fix_head_stem

    self.endpoints = None

    self._build()

  def _get_conv_block(self, conv_type):
    conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
    return conv_block_map[conv_type]

  def _build(self):
    """Builds a model."""
    self._blocks = []

    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon

    if self._global_params.data_format == 'channels_first':
      channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      channel_axis = -1
      self._spatial_dims = [1, 2]

    # Stem part.
    self._conv_stem = utils.Conv2D(
        filters=round_filters(32, self._global_params, self._fix_head_stem),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._global_params.data_format,
        use_bias=False)
    self._bn0 = self._batch_norm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)

    # Builds blocks.
    for i, block_args in enumerate(self._blocks_args):
      assert block_args.num_repeat > 0
      assert block_args.super_pixel in [0, 1, 2]

      # Update block input and output filters based on depth multiplier.
      input_filters = round_filters(block_args.input_filters,
                                    self._global_params)

      output_filters = round_filters(block_args.output_filters,
                                     self._global_params)
      kernel_size = block_args.kernel_size
      if self._fix_head_stem and (i == 0 or i == len(self._blocks_args) - 1):
        repeats = block_args.num_repeat
      else:
        repeats = round_repeats(block_args.num_repeat, self._global_params)
      block_args = block_args._replace(
          input_filters=input_filters,
          output_filters=output_filters,
          num_repeat=repeats)

      # The first block needs to take care of stride and filter size increase.
      conv_block = self._get_conv_block(block_args.conv_type)
      if not block_args.super_pixel:  #  no super_pixel at all
        self._blocks.append(conv_block(block_args, self._global_params))
      else:
        # if superpixel, adjust filters, kernels, and strides.
        depth_factor = int(4 / block_args.strides[0] / block_args.strides[1])
        block_args = block_args._replace(
            input_filters=block_args.input_filters * depth_factor,
            output_filters=block_args.output_filters * depth_factor,
            kernel_size=((block_args.kernel_size + 1) // 2 if depth_factor > 1
                         else block_args.kernel_size))

        # if the first block has stride-2 and super_pixel trandformation
        if (block_args.strides[0] == 2 and block_args.strides[1] == 2):
          block_args = block_args._replace(strides=[1, 1])
          self._blocks.append(conv_block(block_args, self._global_params))
          block_args = block_args._replace(  # sp stops at stride-2
              super_pixel=0,
              input_filters=input_filters,
              output_filters=output_filters,
              kernel_size=kernel_size)
        elif block_args.super_pixel == 1:
          self._blocks.append(conv_block(block_args, self._global_params))
          block_args = block_args._replace(super_pixel=2)
        else:
          self._blocks.append(conv_block(block_args, self._global_params))

      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in xrange(block_args.num_repeat - 1):
        self._blocks.append(conv_block(block_args, self._global_params))

    # Head part.
    self._conv_head = utils.Conv2D(
        filters=round_filters(1280, self._global_params, self._fix_head_stem),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._global_params.data_format,
        use_bias=False)
    self._bn1 = self._batch_norm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)

    self._avg_pooling = nn.AdaptiveAvgPool2d(1)
    self._dropout = nn.Dropout(self._global_params.dropout_rate)
    self._fc = nn.Linear(out_channels, self._global_params.num_classes)
    

