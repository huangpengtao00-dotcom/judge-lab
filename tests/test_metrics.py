"""传统指标复现测试(红线 2 的硬兑现)。

- ΔE2000:Sharma, Wu & Dalal 2005 官方 34 组测试向量全过,容差 1e-4。
  向量原样抄自 Gaurav Sharma 主页 ciede2000testdata.txt
  (http://www2.ece.rochester.edu/~gsharma/ciede2000/,经 Web Archive 取原件),
  含全部不连续点陷阱(hue 均值分支、C'=0 特例、±360° 修正)。
- UCIQE/UIQM 官方无标准测试向量 → 行为契约测试:确定性、干净图>重退化图(多数水体)、
  已知缺陷复现(过饱和/对比拉爆反而得更高分)。
"""

import numpy as np
import pytest

from judgelab.domains.color.metrics import delta_e76, delta_e2000
from judgelab.domains.color.synth import gen_test_image
from judgelab.domains.underwater.metrics import uciqe, uiqm
from judgelab.domains.underwater.synth import apply_water, random_water

# (L1, a1, b1, L2, a2, b2, 期望 ΔE00),Sharma 2005 官方测试数据 34 组,一字未改
CIEDE2000_OFFICIAL = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]


@pytest.mark.parametrize("row", CIEDE2000_OFFICIAL, ids=[f"pair{i+1:02d}" for i in range(34)])
def test_ciede2000_official_vectors(row):
    l1, a1, b1, l2, a2, b2, expected = row
    got = delta_e2000((l1, a1, b1), (l2, a2, b2))
    assert abs(got - expected) <= 1e-4, f"got {got:.6f}, expect {expected}"


def test_ciede2000_vectorized_matches_scalar():
    """(...,3) 批量路径与逐对结果位级一致(判官会批量吃像素)。"""
    arr = np.array(CIEDE2000_OFFICIAL)
    batch = delta_e2000(arr[:, 0:3], arr[:, 3:6])
    scalars = np.array([delta_e2000(r[0:3], r[3:6]) for r in CIEDE2000_OFFICIAL])
    assert np.array_equal(batch, scalars)
    assert np.allclose(batch, arr[:, 6], atol=1e-4)


def test_delta_e76_euclidean():
    # 手算样例:sqrt(2.5² + 2.5²) = 3.5355339…;同点距离为 0
    assert delta_e76((50, 2.5, 0.0), (50, 0.0, -2.5)) == pytest.approx(3.5355339, abs=1e-6)
    assert delta_e76((50, 2.5, 0.0), (50, 2.5, 0.0)) == 0.0
    with pytest.raises(ValueError):
        delta_e76((50, 2.5), (50, 0.0, -2.5))  # 形状不对显式抛(红线1)


# ---------------------------------------------------------------- UCIQE / UIQM 行为契约


def _oversaturate(img: np.ndarray, sat: float = 3.0, con: float = 2.5) -> np.ndarray:
    """过度增强样本:饱和度 ×3 + 对比 ×2.5(裁剪拉爆)。文献批评的典型病态图。"""
    x = img.astype(np.float64) / 255.0
    lum = x @ np.array([0.2126, 0.7152, 0.0722])
    x = lum[..., None] + (x - lum[..., None]) * sat
    x = (x - 0.5) * con + 0.5
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def test_metrics_deterministic():
    """同图同参位级同分(红线 7)。"""
    img = gen_test_image(np.random.default_rng(1))
    assert uciqe(img) == uciqe(img.copy())
    assert uiqm(img) == uiqm(img.copy())


def test_metrics_input_contract():
    """非 uint8 / 非 HxWx3 显式抛 ValueError,不静默降级(红线 1)。"""
    bad_dtype = np.zeros((16, 16, 3), dtype=np.float64)
    bad_shape = np.zeros((16, 16), dtype=np.uint8)
    for bad in (bad_dtype, bad_shape):
        with pytest.raises(ValueError):
            uciqe(bad)
        with pytest.raises(ValueError):
            uiqm(bad)


def test_clean_beats_heavy_degradation_majority():
    """干净图分应高于重退化图(t=1.0),在多数水体下成立。

    只要求多数(>1/2)而非全体:UCIQE/UIQM 本就是弱判官(本仓的被证伪对象),
    个别水体上翻车是预期内行为。实测 seed 0..7 两指标均 8/8。
    """
    n, wins_uciqe, wins_uiqm = 8, 0, 0
    for seed in range(n):
        rng = np.random.default_rng(seed)
        img = gen_test_image(rng)
        degraded = apply_water(img, random_water(rng), 1.0)
        wins_uciqe += uciqe(img) > uciqe(degraded)
        wins_uiqm += uiqm(img) > uiqm(degraded)
    assert wins_uciqe > n / 2, f"uciqe clean>degraded only {wins_uciqe}/{n}"
    assert wins_uiqm > n / 2, f"uiqm clean>degraded only {wins_uiqm}/{n}"


def test_uciqe_rewards_oversaturation():
    """已知缺陷复现:饱和度/对比拉爆的过度增强图,UCIQE 反而给更高分。

    这正是文献对 UCIQE 的核心批评(σ_c、μ_s 随饱和度单调上涨,con_l 随对比
    裁剪上涨,三项全被过度增强抬高)。实测 seed 0..5 全部成立(over > clean),
    差距显著(约 +10,量级为 clean 分的 40%+),故用严格断言。
    """
    for seed in range(6):
        img = gen_test_image(np.random.default_rng(seed))
        over = _oversaturate(img)
        assert uciqe(over) > uciqe(img), f"seed {seed}: oversaturated not rewarded"


def test_uiqm_rewards_oversaturation():
    """同一缺陷在 UIQM 上同样成立:UICM 奖励对立色方差、UISM 奖励边缘比值、
    UIConM 的块对比也被裁剪拉高。实测 seed 0..5 全部成立(约 2 倍于 clean)。"""
    for seed in range(6):
        img = gen_test_image(np.random.default_rng(seed))
        over = _oversaturate(img)
        assert uiqm(over) > uiqm(img), f"seed {seed}: oversaturated not rewarded"
