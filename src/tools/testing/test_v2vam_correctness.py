import torch

from src.models.fuse_modules.V2VAM import V2V_AttFusion, CrissCrossAttention


def _build_model(feature_dim=8):
    model = V2V_AttFusion(feature_dim)
    model.eval()
    return model


def test_v2vam_batch_isolation_group_independence():
    torch.manual_seed(123)
    model = _build_model(feature_dim=8)

    # Two groups, each with 2 CAV features.
    record_len = torch.tensor([2, 2])
    x_ref = torch.randn(4, 8, 6, 7)

    # Keep second group fixed, perturb first group heavily.
    x_alt = x_ref.clone()
    x_alt[:2] = torch.randn_like(x_alt[:2]) * 5.0

    with torch.no_grad():
        out_ref = model(x_ref, record_len)
        out_alt = model(x_alt, record_len)

    # output[1] corresponds to second record_len group and must be independent.
    assert torch.allclose(out_ref[1], out_alt[1], atol=1e-6), "Second group output leaked from first group"


def test_v2vam_group_output_matches_standalone():
    torch.manual_seed(456)
    model = _build_model(feature_dim=8)

    record_len = torch.tensor([2, 2])
    x = torch.randn(4, 8, 6, 7)

    with torch.no_grad():
        out_full = model(x, record_len)
        out_second_only = model(x[2:4], torch.tensor([2]))

    assert torch.allclose(out_full[1:2], out_second_only, atol=1e-6), "Second group output depends on previous groups"


def test_crisscross_cpu_device_safe_forward():
    torch.manual_seed(0)
    cca = CrissCrossAttention(in_dim=8).eval()

    q = torch.randn(1, 8, 5, 6)  # CPU tensor
    with torch.no_grad():
        out = cca(q, q, q)

    assert out.device.type == "cpu", "CPU forward should stay on CPU"
    assert out.shape == q.shape, "Output shape mismatch on CPU"


def main():
    test_v2vam_batch_isolation_group_independence()
    test_v2vam_group_output_matches_standalone()
    test_crisscross_cpu_device_safe_forward()
    print("V2VAM correctness tests passed")


if __name__ == "__main__":
    main()
