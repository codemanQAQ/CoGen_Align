import torch

from cogen_align.models.losses import SymmetricInfoNCE


def test_symmetric_infonce_in_batch():
    b, d = 8, 64
    loss_fn = SymmetricInfoNCE(temperature=0.07)
    z_a = torch.randn(b, d)
    z_t = torch.randn(b, d)
    out = loss_fn(z_a, z_t)
    assert out["loss"].ndim == 0
    assert out["loss"].item() == out["loss"].item()  # finite
    assert 0 <= out["top1_acc"].item() <= 1


def test_symmetric_infonce_hard_negatives():
    b, d, k = 4, 32, 2
    loss_fn = SymmetricInfoNCE(temperature=0.1)
    z_a = torch.randn(b, d)
    z_t = torch.randn(b, d)
    hard = torch.randn(b, k, d)
    out = loss_fn(z_a, z_t, hard_neg_z_t=hard)
    assert out["loss"].shape == ()
    assert not torch.isnan(out["loss"])
