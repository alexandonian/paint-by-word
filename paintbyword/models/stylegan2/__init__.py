####
# This port of styleganv2 is derived from and perfectly compatible with
# the pytorch port by https://github.com/rosinality/stylegan2-pytorch.
#
# In this reimplementation, all non-leaf modules are subclasses of
# nn.Sequential so that the network can be more easily split apart
# for surgery and direct rewriting.

import sys
import os
import pickle
from collections import defaultdict

import torch
from torch.utils import model_zoo

from .models import SeqStyleGAN2, DataBag

# TODO: change these paths to non-antonio paths, probably load from url if not exists
WEIGHT_URLS = 'http://wednesday.csail.mit.edu/placesgan/tracer/utils/stylegan2/weights/'
sizes = defaultdict(lambda: 256, faces=1024, car=512)

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dirname, '../../../lib/stylegan2-ada-pytorch'))


def load_state_dict(category):
    chkpt_name = f'stylegan2_{category}.pt'
    weights_dir = os.path.join(dirname, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    model_path = os.path.join(weights_dir, chkpt_name)

    if not os.path.exists(model_path):
        url = WEIGHT_URLS + chkpt_name
        state_dict = model_zoo.load_url(url, model_dir=weights_dir, progress=True)
        torch.save(state_dict, model_path)
    else:
        state_dict = torch.load(model_path)
    return state_dict


def load_seq_stylegan(category, truncation=1.0, **kwargs):  # mconv='seq'):
    ''' loads nn sequential version of stylegan2 and puts on gpu'''
    state_dict = load_state_dict(category)
    size = sizes[category]
    g = SeqStyleGAN2(size, style_dim=512, n_mlp=8, truncation=truncation, **kwargs)
    g.load_state_dict(state_dict['g_ema'], latent_avg=state_dict['latent_avg'])
    g.cuda()
    return g


def load_stylegan2(pretrained='birds', truncation=1.0, device='cuda'):
    filenames = {
        'birds': 'stylegan2-birds.pkl',
    }
    filename = filenames.get(pretrained, 'birds')
    weights_dir = os.path.join(dirname, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, filename)

    if not os.path.exists(model_path):
        print(f'Downloading weights to: {model_path}')
        weights_url = 'http://pretorched-x.csail.mit.edu/gans/StyleGAN2/'
        cmd = f'wget -O {model_path} {weights_url}{filename}'
        os.system(cmd)

    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module

    f = G.forward

    def forward(z):
        return f(z, None, truncation_psi=truncation)

    G.forward = forward
    G = G.eval()
    G.input_shape = (-1, 512)
    return G


def load_stylegan2_birds(truncation=1.0, device='cuda'):
    return load_stylegan2(pretrained='birds', truncation=truncation, device=device)
