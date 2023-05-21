import jax
from flax.training import checkpoints
import numpy as np
import jax.numpy as jnp

from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import freeze, unfreeze
import flax

def map_to_jax(pytorch_key):
    if 'blocks' in pytorch_key[0]:
        if 'decoder' in pytorch_key[0]:
            jax_key = ('decoder', f'blocks_{pytorch_key[1]}') + pytorch_key[2:]
        else:
            jax_key = ('encoder', f'blocks_{pytorch_key[1]}') + pytorch_key[2:]
    else:
        if pytorch_key[0] == 'decoder_pred' :
            jax_key = ('decoder_image_output', 'Dense_0', *pytorch_key[1:])
#         elif 'patch_embed' == pytorch_key[0]:
#             jax_key = ('image_embedding', *pytorch_key[1:])
        elif 'decoder_embed' == pytorch_key[0]:
            jax_key = ('decoder_input_projection', *pytorch_key[1:])
        elif 'decoder' in pytorch_key[0]:
            jax_key = ('decoder', pytorch_key[0].partition('_')[2], *pytorch_key[1:])
        else:
            if pytorch_key[0] in ['cls_token', 'mask_token', 'patch_embed']:
                jax_key = pytorch_key
            else:
                jax_key = ('encoder', *pytorch_key)
        
    
    if jax_key[-1] == "weight":
        if 'norm' in jax_key[-2]:
            jax_key = jax_key[:-1] + ("scale",)
        else:
            jax_key = jax_key[:-1] + ("kernel",)
    return jax_key


def pytorch_statedict_to_jax(state_dict):
    pytorch_dict = {tuple(k.split('.')): v for k, v in state_dict['model'].items()}
    
    jax_flat_dict = {map_to_jax(k): jnp.asarray(v) for k, v in pytorch_dict.items()}
    for k in jax_flat_dict:
        if k[-1] == 'kernel':
            kernel = jax_flat_dict[k]
            if kernel.ndim > 2: # Conv
                kernel = jnp.transpose(kernel, (2, 3, 1, 0))
            else:
                kernel = jnp.transpose(kernel, (1, 0))
            jax_flat_dict[k] = kernel
    return flax.traverse_util.unflatten_dict(jax_flat_dict)


from absl import app
from absl import flags

from jaxrl_m.vision import encoders
from jaxrl_m.common import TrainState
from flax.training import checkpoints

FLAGS = flags.FLAGS

def load_statedict_from_file(pretrained_path):
  import torch
  return torch.load(pretrained_path, map_location=torch.device('cpu'))


def main(_):
    print(f'Creating encoder_def for {FLAGS.encoder}')
    encoder_def = encoders[FLAGS.encoder]()

    print(f'Loading pretrained weights from {FLAGS.pretrained_path}')
    pytorch_statedict = load_statedict_from_file(FLAGS.pretrained_path)
    restored_params = pytorch_statedict_to_jax(pytorch_statedict)
    restored_ev = flax.core.FrozenDict({})
        
    train_state = TrainState.create(
        model_def=encoder_def,
        params=restored_params,
        tx=None,
        extra_variables=restored_ev,
    )

    new_fname = checkpoints.save_checkpoint(
        ckpt_dir=FLAGS.save_dir,
        target=train_state,
        step=0,
        overwrite=True,
        prefix=FLAGS.prefix+'-',
    )
    print(f'Saved to {new_fname}')

if __name__ == '__main__':
    flags.DEFINE_string('pretrained_path', None, 'Pytorch .pt file ', required=True)
    flags.DEFINE_string('save_dir', 'pretrained_encoders_new', 'Where to save parameter file.')
    flags.DEFINE_string('prefix', None, 'Where to save parameter file.', required=True)
    flags.DEFINE_string('encoder', 'mae_base', 'Which encoder?')
    app.run(main)