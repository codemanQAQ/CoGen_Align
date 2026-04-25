import torch

from cogen_align.models.projector import AttentionPooling, Projector, SpeechTextEncoder


def test_projector_subsample_and_pool():
    b, t, d = 2, 25, 1280
    x = torch.randn(b, t, d)
    proj = Projector(in_dim=d, out_dim=64, hidden_dim=128, subsample=5)
    y = proj(x)
    assert y.shape[0] == b
    assert y.shape[-1] == 64

    emb = torch.nn.Embedding(1000, 64)
    enc = SpeechTextEncoder(proj, emb, out_dim=64)
    z_a = enc.encode_audio(x, audio_mask=None)
    assert z_a.shape == (b, 64)
    tid = torch.randint(0, 1000, (b, 16))
    tmask = torch.zeros(b, 16, dtype=torch.bool)
    z_t = enc.encode_text(tid, tmask)
    assert z_t.shape == (b, 64)

    pool = AttentionPooling(32)
    z = pool(torch.randn(3, 10, 32))
    assert z.shape == (3, 32)
