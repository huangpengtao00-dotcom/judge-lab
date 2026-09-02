"""水下合成器契约测试:恒等/确定性/物理方向性(红通道最先死、越深越退化)。"""

import numpy as np

from judgelab.core.evidence import srgb_to_lab
from judgelab.domains.color.synth import gen_test_image
from judgelab.domains.underwater.synth import apply_water, gen_dataset, random_water


def test_identity_and_determinism():
    rng = np.random.default_rng(3)
    img = gen_test_image(rng)
    water = random_water(rng)
    assert np.array_equal(apply_water(img, water, 0.0), img)
    a, b = apply_water(img, water, 0.6), apply_water(img, water, 0.6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, img)


def test_physics_direction():
    """红衰减 > 蓝衰减;退化随 t 单调加深(以红通道均值与整体亮度为探针)。"""
    rng = np.random.default_rng(3)
    img = gen_test_image(rng)
    water = random_water(rng)
    red_means, lums = [], []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = apply_water(img, water, t).astype(np.float64)
        red_means.append(out[..., 0].mean())
        lums.append(srgb_to_lab(out.astype(np.uint8))[..., 0].mean())
    # 红通道均值单调下降(衰减+雾幕里红分量低)
    assert all(red_means[i] >= red_means[i + 1] for i in range(4)), red_means
    # 亮度单调下降(低照分量)
    assert all(lums[i] >= lums[i + 1] for i in range(4)), lums


def test_dataset_contract(tmp_path):
    manifest = gen_dataset(tmp_path / "uw", n_images=2, n_waters=2, seed=0)
    lines = manifest.read_text().splitlines()
    assert len(lines) == 2 * 2 * 5
    import json

    row = json.loads(lines[0])
    assert set(row) >= {"case_id", "source", "result", "gt_strength", "group", "water"}
    assert "reference" not in row  # 水下契约:无参考图,评估器吃 (source, result) 或仅 result
