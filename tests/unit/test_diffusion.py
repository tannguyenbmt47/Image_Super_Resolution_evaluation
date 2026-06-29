import pytest
import torch

from src.models import build_model
from src.models.diffusion import GaussianDiffusion, UNet, pad_to_multiple
from src.models.diffusion.gaussian_diffusion import make_beta_schedule
from src.utils import Config


def test_pad_to_multiple():
    p, (h, w) = pad_to_multiple(torch.rand(1, 3, 7, 5), 8)
    assert p.shape[-2:] == (8, 8)
    assert (h, w) == (7, 5)


def test_unet_forward_shape():
    net = UNet(in_channels=6, out_channels=3, base_channels=8,
               channel_mults=[1, 2], num_res_blocks=1, attn_levels=[1])
    out = net(torch.rand(2, 6, 16, 16), torch.randint(0, 10, (2,)))
    assert out.shape == (2, 3, 16, 16)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_beta_schedule_valid(schedule):
    betas = make_beta_schedule(schedule, 50)
    assert betas.shape == (50,)
    assert (betas > 0).all() and (betas < 1).all()


def test_beta_schedule_unknown_raises():
    with pytest.raises(ValueError):
        make_beta_schedule("nope", 10)


def test_q_sample_shape():
    gd = GaussianDiffusion(lambda x, t, c: x[:, :3], timesteps=20)
    x0 = torch.rand(2, 3, 16, 16)
    t = torch.randint(0, 20, (2,))
    assert gd.q_sample(x0, t, torch.randn_like(x0)).shape == x0.shape


@pytest.mark.parametrize("name", ["SR3"])
def test_diffusion_loss_and_sample(name, tiny_diffusion_args):
    m = build_model(Config({"name": name, "args": {"scale": 4, **tiny_diffusion_args}}))
    lr, hr = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 32, 32)

    loss = m.compute_loss(lr, hr)
    loss.backward()  # gradients must flow
    assert loss.item() >= 0

    sr = m.super_resolve(lr)
    assert sr.shape == (2, 3, 32, 32)
    assert sr.min() >= 0 and sr.max() <= 1


def test_diffusion_handles_non_divisible_size(tiny_diffusion_args):
    m = build_model(Config({"name": "SR3", "args": {"scale": 4, **tiny_diffusion_args}}))
    sr = m.super_resolve(torch.rand(1, 3, 7, 5))
    assert sr.shape == (1, 3, 28, 20)
