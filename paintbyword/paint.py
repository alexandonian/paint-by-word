# Method class will
# - take a stylegan model and modify it in order to support the method
# - instantiate and hold a clip model, and also a set of standard z seeds.
# - given an image's z, a mask and a text, derives a new w to insert the
#    requested concept.   Can optionally provide feedback during this process.
import contextlib
import copy
import os

import torch

from paintbyword import cma_optim, losses
from paintbyword.models.biggan import BigGAN
from paintbyword.models.stylegan2 import DataBag, load_seq_stylegan, load_stylegan2
from paintbyword.utils import (
    labwidget,
    nethook,
    paintwidget,
    pbar,
    renormalize,
    show,
    smoothing,
    zdataset,
)

DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'


class BasePainter:
    def sample_fixed_z(self, first_i=0, batch_size=1):
        '''
        Deterministic sampling of z.
        '''
        return torch.stack([self.zds[first_i + i][0] for i in range(batch_size)]).to(
            self.device
        )

    def unpainted(self, initial_z):
        if isinstance(initial_z, int):
            initial_z = self.sample_fixed_z(initial_z)
        return self.gan_model(initial_z)

    def load_text_features(self, description):
        self.description = description
        self.text_features = self.clip_loss.init_clip_target(description)

    def load_mask_and_bounds(self, mask_url):
        if mask_url:
            self.im_mask = renormalize.from_url(
                mask_url, target='pt', size=self.image_shape[2:]
            )[0].to(self.device)
        else:
            self.im_mask = torch.ones(self.image_shape[2:]).to(self.device)
        self.bounds = square_bounding_box(self.im_mask)

    def reconstruction_loss(self, output):
        return self.proj_loss(output, self.original_im, weight=(1 - self.im_mask[None][None]))

    def full_clip_loss(self, images, mode='paste_onto_blank'):
        if mode == 'zoom':
            images = upsample_bb(images, self.bounds, self.clip_loss.clip_res)
        elif mode == 'paste_onto_blank':
            images = self.loss_mask * images
        elif mode == 'paste_onto_orig':
            images = self.original_im * (1 - self.loss_mask) + images * self.loss_mask
        elif mode == 'paste_onto_blank_full_orig':
            return (self.clip_loss(self.loss_mask * images) + self.clip_loss(images)) / 2
        return self.clip_loss(images)

    def full_loss(self, out):
        total_loss = self.full_clip_loss(out, mode=self.clip_loss_mode)
        if self.proj_loss_lambda > 0.0 and len((1 - self.loss_mask).nonzero()) > 0:
            total_loss += self.proj_loss_lambda * self.reconstruction_loss(out)
        return total_loss

    def paint(self, initial_z, description, mask_url=None, optim_var='w', **kwargs):
        raise NotImplementedError()

    def optimize_cma(
        self,
        model,
        optim_var,
        seed=1,
        sigma0=0.5,
        pop_size=50,
        cma_adapt=True,
        cma_diag=False,
        cma_active=True,
        cma_elitist=False,
        num_iterations=20,
        ema_decay=0.5,
        proj_loss_lambda=5.0,
        clip_loss_mode='paste_onto_blank',
    ):

        self.clip_loss_mode = clip_loss_mode
        self.proj_loss_lambda = proj_loss_lambda
        self.smoothed_mask = blur_mask(self.im_mask.clone())
        self.loss_mask = self.im_mask

        self.cma_opts = {
            'popsize': pop_size,
            'seed': seed,
            'AdaptSigma': cma_adapt,
            'CMA_diagonal': cma_diag,
            'CMA_active': cma_active,
            'CMA_elitist': cma_elitist,
            'bounds': None,
        }

        best_loss = None
        all_ims, all_losses = [], []
        best_var = optim_var[0][None].clone()
        cmaes = cma_optim.CMA(optim_var, sigma0, **self.cma_opts)

        for _ in pbar(range(num_iterations)):
            with torch.no_grad():
                cmaes.step()

                im = model(optim_var)
                loss = self.full_loss(im)
                l_min, l_idx = loss.min(dim=0)

                all_losses.append(l_min.cpu())
                all_ims.append(im[l_idx].cpu())
                if best_loss is None or l_min.item() < best_loss:
                    best_loss = l_min.item()
                    bz = optim_var[l_idx][None].clone()
                    best_var = best_var * ema_decay + (1 - ema_decay) * bz
                cmaes.backward(loss.tolist())
        return best_var, all_losses, all_ims

    def optimize_sgd(
        self,
        model,
        optim_var,
        learning_rate=0.01,
        num_iterations=100,
        loss_history=None,
        im_history=None,
    ):
        optim_var.requires_grad = True
        im_history = [] if im_history is None else im_history
        loss_history = [] if loss_history is None else loss_history
        optimizer = torch.optim.Adam([optim_var], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, learning_rate, num_iterations
        )
        for _ in pbar(range(num_iterations)):
            im = model(optim_var)
            loss = self.full_loss(im)

            im_history.append(im[0].detach().cpu())
            loss_history.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        return optim_var, loss_history, im_history

    def show_samples(self, z):
        with torch.no_grad():
            if isinstance(z, torch.Tensor):
                show([[pilim(x), i] for i, x in enumerate(self.unpainted(z))])
            elif isinstance(z, torch.utils.data.dataset.TensorDataset):
                show(
                    [
                        [pilim(self.unpainted(z[0][None].to(self.device))[0][0]), i]
                        for i, z in enumerate(z)
                    ]
                )

    def get_scribble_mask_url(self, z):
        widget_array = [paintwidget.PaintWidget(image=renormalize.as_url(self.unpainted(z)[0]))]

        def do_reset():
            for pw in widget_array:
                pw.mask = ''

        reset_btn = labwidget.Button('reset').on('click', do_reset)
        widget_array[0].mask = ''
        if labwidget.WIDGET_ENV is not None:
            show([widget_array, reset_btn])
        self.widget_array = widget_array

    def show_seed_images(self, batch_size=32, seed=None, class_label=None):
        with torch.no_grad():
            self.show_samples(self.sample_fixed_z(batch_size=batch_size))

    def mask_seed_image(self, choice):
        with torch.no_grad():
            initial_z = self.sample_fixed_z(choice)
            self.get_scribble_mask_url(initial_z)

    def get_mask_url(self):
        return self.widget_array[0].mask


class StyleganPainter(BasePainter):
    def __init__(
        self,
        pretrained='birds',
        sample_size=2000,
        num_clip_crops=32,
        truncation=1.0,
        device='cuda',
    ):
        self.gan_model = load_stylegan2(
            pretrained=pretrained,
            truncation=truncation,
            device=device,
        )

        self.device = device
        self.truncation = truncation
        self.sample_size = sample_size
        self.num_clip_crops = num_clip_crops

        self.clip_loss = losses.CLIPLoss('', reduction='none', num_crops=num_clip_crops)
        self.proj_loss = losses.ProjectionLoss()
        nethook.set_requires_grad(False, self.gan_model)
        nethook.set_requires_grad(False, self.clip_loss)
        nethook.set_requires_grad(False, self.proj_loss)

        self.zds = zdataset.z_dataset_for_model(self.gan_model, size=sample_size)
        with torch.no_grad():
            sample_z = self.sample_fixed_z()
            sample_image = self.gan_model(sample_z)
        self.image_shape = sample_image.shape

    def load_model(self, initial_z, mask_url):
        if isinstance(initial_z, int):
            initial_z = self.sample_fixed_z(initial_z)
        self.initial_z = initial_z
        self.mask_url = mask_url

    def paint(self, initial_z, description, mask_url=None, optim_method='cma', **kwargs):
        self.load_text_features(description)
        self.load_mask_and_bounds(mask_url)
        self.load_model(initial_z, mask_url)
        return self.optimize_image_to_text(optim_method=optim_method, **kwargs)

    def optimize_image_to_text(
        self,
        optim_method='cma',
        learning_rate=0.01,
        num_iterations=100,
        **kwargs,
    ):
        with torch.no_grad():
            z = self.initial_z.clone()
            self.original_im = self.gan_model(z)

        if 'cma' in optim_method.lower():
            with torch.no_grad():
                z, loss_history, im_history = self.optimize_cma(self.gan_model, z)
        if 'adam' in optim_method.lower() or 'sgd' in optim_method.lower():
            z, loss_history, im_history = self.optimize_sgd(
                self.gan_model,
                z,
                learning_rate=learning_rate,
                num_iterations=num_iterations,
                loss_history=loss_history,
                im_history=im_history,
            )

        with torch.no_grad():
            return z, self.gan_model(z), loss_history, im_history


class StyleganMaskedPainter(BasePainter):
    def __init__(
        self,
        pretrained='bedroom',
        sample_size=2000,
        num_clip_crops=32,
        truncation=1.0,
        device='cuda',
    ):

        self.gan_model = (
            load_seq_stylegan(pretrained, mconv='seq', truncation=truncation).eval().to(device)
        )

        self.device = device
        self.pretrained = pretrained
        self.truncation = truncation
        self.sample_size = sample_size
        self.num_clip_crops = num_clip_crops

        self.clip_loss = losses.CLIPLoss('', reduction='none', num_crops=num_clip_crops)
        self.proj_loss = losses.ProjectionLoss()
        nethook.set_requires_grad(False, self.gan_model)
        nethook.set_requires_grad(False, self.clip_loss)
        nethook.set_requires_grad(False, self.proj_loss)

        self.zds = zdataset.z_dataset_for_model(self.gan_model, size=sample_size)
        with torch.no_grad():
            sample_z = self.sample_fixed_z()
            with nethook.Trace(self.gan_model, 'latents') as tr:
                sample_image = self.gan_model(sample_z)
                sample_wplus = tr.output.latent
                self.wplus_shape = sample_wplus.shape
        self.image_shape = sample_image.shape

    def paint(
        self,
        initial_z,
        description,
        mask_url=None,
        optim_method='cma',
        optim_var='w',
        **kwargs,
    ):
        self.load_text_features(description)
        self.load_mask_and_bounds(mask_url)
        self.load_masked_model(initial_z, mask_url)
        return self.optimize_image_to_text(
            optim_method=optim_method, optim_var=optim_var, **kwargs
        )

    def load_mask_and_bounds(self, mask_url):
        mask_url = self.get_mask_url() if mask_url is None else mask_url
        if mask_url:
            self.im_mask = renormalize.from_url(
                mask_url, target='pt', size=self.image_shape[2:]
            )[0].to(self.device)
        else:
            self.im_mask = torch.ones(self.image_shape[2:]).to(self.device)
        self.bounds = square_bounding_box(self.im_mask)

    def load_masked_model(self, initial_z, mask_url, **kwargs):
        mask_url = self.get_mask_url() if mask_url is None else mask_url
        if isinstance(initial_z, int):
            initial_z = self.sample_fixed_z(initial_z)
        self.initial_z = initial_z
        self.mask_url = mask_url
        masked_model = make_masked_stylegan(self.gan_model, initial_z, self.mask_url, **kwargs)
        # Now split the model to allow direct input of w.
        first_half = nethook.subsequence(masked_model, last_layer='latents')
        second_half = nethook.subsequence(masked_model, after_layer='latents')
        self.w_model = nethook.concatenate_sequence(
            first_half, ('return_w', ReturnLatentW())
        ).eval()
        self.masked_model = nethook.concatenate_sequence(
            ('input_w', InputLatentWtoWPlus(self.wplus_shape)), second_half
        ).eval()

    def optimize_image_to_text(
        self,
        optim_method='cma',
        optim_var='w',
        learning_rate=0.01,
        num_iterations=100,
        **kwargs,
    ):
        with torch.no_grad():
            z = self.initial_z.clone()
            self.original_im = self.gan_model(z)

        var = self.w_model(z).clone() if optim_var == 'w' else z
        model = (
            self.masked_model
            if optim_var == 'w'
            else lambda z: self.masked_model(self.w_model(z))
        )

        if 'cma' in optim_method.lower():
            with torch.no_grad():
                var, loss_history, im_history = self.optimize_cma(model, var)
        if 'adam' in optim_method.lower() or 'sgd' in optim_method.lower():
            var, loss_history, im_history = self.optimize_sgd(
                model,
                var,
                learning_rate=learning_rate,
                num_iterations=num_iterations,
                loss_history=loss_history,
                im_history=im_history,
            )

        with torch.no_grad():
            return var, model(var), loss_history, im_history


class ReturnLatentW(torch.nn.Module):
    def forward(self, d):
        return d.latent[:, 0, :]


class InputLatentWtoWPlus(torch.nn.Module):
    def __init__(self, wplus_shape):
        super().__init__()
        self.wplus_shape = wplus_shape

    def forward(self, w):
        wplus = w[:, None, :].expand(-1, self.wplus_shape[1], -1)
        return DataBag(latent=wplus)


class ApplyMaskedStyle(torch.nn.Module):
    def __init__(self, mask, fixed_style):
        super().__init__()
        self.register_buffer('mask', mask)
        self.register_buffer('fixed_style', fixed_style)

    def forward(self, d):
        modulation = (d.style[:, :, None, None] * self.mask) + (
            self.fixed_style[:, :, None, None] * (1 - self.mask)
        )
        return DataBag(
            d,
            fmap=modulation * d.fmap,
            style=self.fixed_style.expand(d.fmap.shape[0], -1),
        )


def make_masked_stylegan(
    gan_model, initial_z, mask_url, start_layer=4, middle_layer=9, end_layer=11
):
    '''
    Given a stylegan and a mask (encoded as a PNG) and an initial z,
    creates a modified stylegan which applies z only to a masked
    region.
    '''
    with torch.no_grad():
        style_layers = [n for n, _ in gan_model.named_modules() if 'adain' in n]
        with contextlib.ExitStack() as stack:
            retained_inputs = {
                layer: stack.enter_context(nethook.Trace(gan_model, layer, retain_input=True))
                for layer in style_layers
            }
            gan_model(initial_z)
            style_vectors = {
                layer: retained_inputs[layer].input.style for layer in style_layers
            }
            style_shapes = {
                layer: retained_inputs[layer].output.fmap.shape for layer in style_layers
            }
        masked_model = copy.deepcopy(gan_model)
        device = next(masked_model.parameters()).device
        if mask_url is None:
            return masked_model
        for layer in style_layers:
            parent = nethook.get_module(masked_model, layer[: -len('.adain')])
            vec = style_vectors[layer].to(device)
            shape = style_shapes[layer][-2:]
            mask = renormalize.from_url(mask_url, target='pt', size=shape)[0]
            mask = mask[None, None]
            if shape[0] > 16:
                sigma = float(shape[0]) / 16.0
                kernel_size = int(sigma) * 2 - 1
                blur = smoothing.GaussianSmoothing(1, kernel_size, sigma=sigma)
                mask = blur(mask)
            mask = mask[0, 0].to(device)
            parent.adain = ApplyMaskedStyle(mask, vec)
    return masked_model


def blur_mask(mask):
    shape = mask.shape[-2:]
    sigma = float(shape[0]) / 16.0
    kernel_size = int(sigma) * 2 - 1
    blur = smoothing.GaussianSmoothing(1, kernel_size, sigma=sigma)
    mask = blur(mask[None][None].cpu()).to(mask.device)
    return mask


def bounding_box(mask):
    bmask = mask.bool()
    nz0 = bmask.any(dim=1).nonzero(as_tuple=False)
    nz1 = bmask.any(dim=0).nonzero(as_tuple=False)
    if len(nz0) == 0:
        return [0, bmask.shape[0], 0, bmask.shape[1]]
    return [i.item() for i in [nz0[0], nz0[-1] + 1, nz1[0], nz1[-1] + 1]]


def expand_range(low, high, size, minlim, maxlim):
    excess = size - high + low
    low = min(max(low - excess // 2, minlim), maxlim - size)
    return low, low + size


def square_bounding_box(mask, min_size=128):
    bb = bounding_box(mask)
    sizes = [(bb[i + 1] - bb[i]) for i in [0, 2]]
    target = max(sizes + [min_size])
    for i, s in zip([0, 2], sizes):
        bb[i], bb[i + 1] = expand_range(bb[i], bb[i + 1], target, 0, mask.shape[i // 2])
    return bb


def upsample_bb(data, bb, target_size):
    return torch.nn.functional.interpolate(
        data[:, :, bb[0] : bb[1], bb[2] : bb[3]],
        size=target_size,
        mode='bilinear',
        align_corners=False,
    )


def layer_regularizer(layer, orig):
    return ((layer - orig)).pow(2).sum() / (layer.size(2) * layer.size(3))


def pilim(idata):
    return renormalize.as_image(idata)
