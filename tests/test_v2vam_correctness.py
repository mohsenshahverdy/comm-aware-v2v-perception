from src.tools.testing import test_v2vam_correctness


def test_batch_isolation_group_independence():
    test_v2vam_correctness.test_v2vam_batch_isolation_group_independence()


def test_group_output_matches_standalone():
    test_v2vam_correctness.test_v2vam_group_output_matches_standalone()


def test_cpu_device_safe_forward():
    test_v2vam_correctness.test_crisscross_cpu_device_safe_forward()
