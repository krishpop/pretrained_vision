# Copyright 2022 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BiT models as in the paper (ResNet V2) w/ loading of public weights.

See reproduction proof: http://(internal link)
"""

import functools
import re
from typing import Optional, Sequence, Union

from pretrained_vision import bigvision_utils as u
from pretrained_vision import bigvision_common as common
import flax.linen as nn
import jax.numpy as jnp
import jax

def load(init_params, init_file, model_cfg, dont_load=()):
  """Loads the TF-dumped NumPy or big_vision checkpoint.

  Args:
    init_params: random init params from which the new head is taken.
    init_file: comes from `config.model_init`, can either be an absolute
      path (ie starts with /) to the checkpoint, or a string like
      "L-imagenet2012" describing one of the variants from the paper.
    model_cfg: the model configuration.
    dont_load: list of param names to be reset to init.

  Returns:
    The loaded parameters.
  """

  # Support for vanity model names from the paper.
  vanity = {
      'FunMatch-224px-i1k82.8': 'gs://bit_models/distill/R50x1_224.npz',
      'FunMatch-160px-i1k80.5': 'gs://bit_models/distill/R50x1_160.npz',
  }
  if init_file[0] in ('L', 'M', 'S'):  # The models from the original paper.
    # Supported names are of the following type:
    # - 'M' or 'S': the original "upstream" model without fine-tuning.
    # - 'M-ILSVRC2012': i21k model fine-tuned on i1k.
    # - 'M-run0-caltech101': i21k model fine-tuned on VTAB's caltech101.
    #    each VTAB fine-tuning was run 3x, so there's run0, run1, run2.
    if '-' in init_file:
      up, down = init_file[0], init_file[1:]
    else:
      up, down = init_file, ''
    down = {'-imagenet2012': '-ILSVRC2012'}.get(down, down)  # normalize
    fname = f'BiT-{up}-R{model_cfg.depth}x{model_cfg.width}{down}.npz'
    fname = f'gs://bit_models/{fname}'
  else:
    fname = vanity.get(init_file, init_file)

  params = u.load_params(None, fname)
  params = maybe_convert_big_transfer_format(params)
  return common.merge_params(params, init_params, dont_load)


def maybe_convert_big_transfer_format(params_tf):
  """If the checkpoint comes from legacy codebase, convert it."""

  # Only do anything at all if we recognize the format.
  if 'resnet' not in params_tf:
    return params_tf

  # For ease of processing and backwards compatibility, flatten again:
  params_tf = dict(u.tree_flatten_with_names(params_tf)[0])

  # Works around some files containing weird naming of variables:
  for k in list(params_tf):
    k2 = re.sub('/standardized_conv2d_\\d+/', '/standardized_conv2d/', k)
    if k2 != k:
      params_tf[k2] = params_tf[k]
      del params_tf[k]

  params = {
      'root_block': {'conv_root': {'kernel': params_tf[
          'resnet/root_block/standardized_conv2d/kernel']}},
      'norm-pre-head': {
          'bias': params_tf['resnet/group_norm/beta'][None, None, None],
          'scale': params_tf['resnet/group_norm/gamma'][None, None, None],
      },
      'head': {
          'kernel': params_tf['resnet/head/conv2d/kernel'][0, 0],
          'bias': params_tf['resnet/head/conv2d/bias'],
      }
  }

  for block in ('block1', 'block2', 'block3', 'block4'):
    params[block] = {}
    units = set([re.findall(r'unit\d+', p)[0] for p in params_tf.keys()
                 if p.find(block) >= 0])
    for unit in units:
      params[block][unit] = {}
      for i, group in enumerate('abc', 1):
        params[block][unit][f'conv{i}'] = {
            'kernel': params_tf[f'resnet/{block}/{unit}/{group}/standardized_conv2d/kernel']  # pylint: disable=line-too-long
        }
        params[block][unit][f'gn{i}'] = {
            'bias': params_tf[f'resnet/{block}/{unit}/{group}/group_norm/beta'][None, None, None],  # pylint: disable=line-too-long
            'scale': params_tf[f'resnet/{block}/{unit}/{group}/group_norm/gamma'][None, None, None],  # pylint: disable=line-too-long
        }

      projs = [p for p in params_tf.keys()
               if p.find(f'{block}/{unit}/a/proj') >= 0]
      assert len(projs) <= 1
      if projs:
        params[block][unit]['conv_proj'] = {
            'kernel': params_tf[projs[0]]
        }

  return params


from absl import app
from absl import flags

from jaxrl_m.vision import encoders
from jaxrl_m.common import TrainState
from flax.training import checkpoints

FLAGS = flags.FLAGS

def main(_):
  print(f'Creating encoder_def for {FLAGS.encoder}')
  encoder_def = encoders[FLAGS.encoder]()
  print(f'Creating init_params for {FLAGS.encoder}')
  init_ev, init_params = encoder_def.init(jax.random.PRNGKey(0), jnp.ones([1, 224, 224, 3]), train=True).pop('params')

  print(f'Loading pretrained weights from {FLAGS.pretrained_path}')
  restored_params = load(init_params, FLAGS.pretrained_path, None, dont_load=('head/bias', 'head/kernel'))

  train_state = TrainState.create(
    model_def=encoder_def,
    params=restored_params,
    tx=None,
    extra_variables=init_ev,
  )

  new_fname = checkpoints.save_checkpoint(
    ckpt_dir=FLAGS.save_dir,
    target=train_state,
    step=0,
    overwrite=True,
    prefix=FLAGS.prefix,
  )
  print(f'Saved to {new_fname}')

if __name__ == '__main__':
    flags.DEFINE_string('pretrained_path', None, 'Pretrained thing to load.', required=True)
    flags.DEFINE_string('save_dir', 'pretrained_encoders_new', 'Where to save parameter file.')
    flags.DEFINE_string('prefix', None, 'Where to save parameter file.', required=True)
    flags.DEFINE_string('encoder', 'resnetv2-50-1', 'Which encoder?')
    app.run(main)