import jax
from flax.training import checkpoints
import numpy as np
import jax.numpy as jnp

# function to translate pytorch param keys to jax:
def translate_key(pytorch_name, resnet="50"):
    if resnet == "50":
      block_name = "BottleneckResNetBlock" 
      layer_list = [3, 4, 6, 3]
    elif resnet == "18":
      block_name = "ResNetBlock"
      layer_list = [2, 2, 2, 2]
    else:
      raise RuntimeError("Choose one of {'18', '50'}.")

    split = pytorch_name.split('.')
    
    # fc.{weight|bias} -> (params, Dense_0, {kernel|bias})
    if len(split) == 2 and split[0] == 'fc':
      return ("params", "Dense_0", "bias" if split[1] == "bias" else "kernel")

    # layer{i}.{j}.bn{k}.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, BottleneckBlock_{}, BatchNorm_{}, {scale|bias|mean|var})
    if len(split) == 4 and split[0][:-1] == 'layer' and split[1].isdigit() and split[2][:-1] == 'bn':
      if split[3] in ['num_batches_tracked']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params" if split[3] in ["weight", "bias"] else "batch_stats",
              f"{block_name}_{sum(layer_list[:int(split[0][-1]) - 1]) + int(split[1])}",
              f"BatchNorm_{int(split[2][-1]) - 1}",
              "scale" if split[3] == "weight" else split[3][8:] if split[3] in ["running_mean", "running_var"] else "bias")

    # layer{i}.{j}.conv{k}.weight -> (params, BottleneckBlock_{}, Conv_{}, kernel)
    if len(split) == 4 and split[0][:-1] == 'layer' and split[1].isdigit() and split[2][:-1] == 'conv':
      if split[3] in ['bias']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params",
              f"{block_name}_{sum(layer_list[:int(split[0][-1]) - 1]) + int(split[1])}",
              f"Conv_{int(split[2][-1]) - 1}",
              "kernel")

    # bn1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, bn_init, {scale|bias|mean|var})
    if len(split) == 2 and split[0] == "bn1":
      if split[1] in ['num_batches_tracked']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params" if split[1] in ["weight", "bias"] else "batch_stats",
              "bn_init",
              "scale" if split[1] == "weight" else split[1][8:] if split[1] in ["running_mean", "running_var"] else "bias")

    # conv1.weight -> (params, conv_init, kernel)
    if len(split) == 2 and split[0] == "conv1":
      if split[1] in ['bias']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params", "conv_init", "kernel")

    # layer{i}.{j}.downsample.0.weight -> (params, BottleneckBlock_{}, conv_proj, kernel)
    if len(split) == 5 and split[0][:-1] == 'layer' and split[1].isdigit() and split[2] == 'downsample' and split[3] == '0':
      if split[4] in ['bias']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params",
              f"{block_name}_{sum(layer_list[:int(split[0][-1]) - 1]) + int(split[1])}",
              "conv_proj",
              "kernel")

    # layer{i}.{j}.downsample.1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, BottleneckBlock_{}, norm_proj, {scale|bias|mean|var})
    if len(split) == 5 and split[0][:-1] == 'layer' and split[1].isdigit() and split[2] == 'downsample' and split[3] == '1':
      if split[4] in ['num_batches_tracked']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params" if split[4] in ["weight", "bias"] else "batch_stats",
              f"{block_name}_{sum(layer_list[:int(split[0][-1]) - 1]) + int(split[1])}",
              "norm_proj",
              "scale" if split[4] == "weight" else split[4][8:] if split[4] in ["running_mean", "running_var"] else "bias")


    print(f"NO PATTERN MATCHES: {pytorch_name}")
    return None

def convert_pytorch_to_jax(pytorch_statedict, jax_variables, resnet_type="50"):
  from flax.traverse_util import flatten_dict, unflatten_dict
  from flax.core import freeze, unfreeze


  jax_params = flatten_dict(unfreeze(jax_variables))
  # create a new dict the same shape as the original jax params dict but filled with the (transposed) pytorch weights
  jax2pytorch = {translate_key(key, resnet_type): key for key in pytorch_statedict.keys() if translate_key(key, resnet_type) is not None}
  pytorch_params = {k: v.numpy().T if len(v.shape) != 4 else v.numpy().transpose((2, 3, 1, 0))
                    for k, v in pytorch_statedict.items()}
  new_jax_params = freeze(unflatten_dict({key: pytorch_params[jax2pytorch[key]] for key in jax_params.keys()}))
  return new_jax_params


from absl import app
from absl import flags

from jaxrl_m.vision import encoders
from jaxrl_m.common import TrainState
from flax.training import checkpoints

FLAGS = flags.FLAGS

def load_r3m_statedict(pretrained_path):
    from r3m import load_r3m, load_r3m_reproduce
    assert pretrained_path in ["r3m"]
    r3m = load_r3m_reproduce(pretrained_path)
    r3m.eval(); r3m.cpu();
    r3m_statedict = {k.split('module.convnet.')[1]: v for k, v in r3m.state_dict().items() if 'module.convnet' in k}
    return r3m_statedict

def load_vip_statedict(pretrained_path):
    import vip
    vip_model = vip.load_vip()
    # vip_model.module.convnet.fc = torch.nn.Identity()
    vip_model.eval(); vip_model.cpu();
    vip_statedict = {k.split('module.convnet.')[1]: v for k, v in vip_model.state_dict().items() if 'module.convnet' in k}
    return vip_statedict

def load_imagenet_statedict(pretrained_path, encoder):
    import torch
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    import torchvision

    if encoder == 'resnetv1-50':
        model = torchvision.models.resnet50(pretrained=True)
    elif encoder == "resnetv1-18":
        model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.state_dict()

def load_statedict_from_file(pretrained_path):
  import torch
  return torch.load(pretrained_path, map_location=torch.device('cpu'))


def main(_):
    print(f'Creating encoder_def for {FLAGS.encoder}')
    encoder_def = encoders[FLAGS.encoder]()
    print(f'Creating init_params for {FLAGS.encoder}')
    init_params_and_ev = encoder_def.init(jax.random.PRNGKey(0), jnp.ones([1, 224, 224, 3]), train=True)

    print(f'Loading pretrained weights from {FLAGS.pretrained_path}')
    if '.pt' in FLAGS.pretrained_path:
        pytorch_statedict = load_statedict_from_file(FLAGS.pretrained_path)
    elif 'r3m' in FLAGS.pretrained_path:
        pytorch_statedict = load_r3m_statedict(FLAGS.pretrained_path)
    elif 'vip' in FLAGS.pretrained_path:
        pytorch_statedict = load_vip_statedict(FLAGS.pretrained_path)
    elif 'imagenet' in FLAGS.pretrained_path:
        pytorch_statedict = load_imagenet_statedict(FLAGS.pretrained_path, FLAGS.encoder)
    
    new_params_and_ev = convert_pytorch_to_jax(pytorch_statedict, init_params_and_ev, FLAGS.encoder.split('-')[1])
    restored_ev, restored_params = new_params_and_ev.pop('params')
        
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
    flags.DEFINE_string('pretrained_path', None, 'Choose between imagenet, r3m, or some resnet statedict .pt file ', required=True)
    flags.DEFINE_string('save_dir', 'pretrained_encoders_new', 'Where to save parameter file.')
    flags.DEFINE_string('prefix', None, 'Where to save parameter file.', required=True)
    flags.DEFINE_string('encoder', 'resnetv1-50', 'Which encoder?')
    app.run(main)
