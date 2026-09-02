"""参数化水下退化合成:造"退化程度已知"的图像对(水下评估器的可验证数据)。

物理直觉(简化的水下成像模型,Jaffe-McGlamery 一族的常用近似):
  I(x) = J(x) * t_c(x) + B_c * (1 - t_c(x))
  —— 直射光按通道衰减(红光最先被吃掉),剩下的被水体背散射的"雾幕"填充。
本合成器把它参数化成 Water(衰减系数/水色/雾幕强度/低照),以强度 t ∈ [0,1] 施加:
  result = degrade(source, water, t),t 即"退化程度"的 ground truth。
评估器对 result 打的"感知质量"分应随 t 单调下降 —— 零标注的可验证判据,
与 color 域的 Look/t 完全同构(manifest 契约一致:source=干净图,result=退化图,无 reference)。

注意(红线 2):本文件只管造数据;UIQM/UCIQE 等被证伪对象的复现必须按原论文公式
另立文件并附测试样例,禁止在这里顺手自写。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..color.synth import gen_test_image


@dataclass(frozen=True)
class Water:
    # 通道衰减系数(红 > 绿 > 蓝 是典型海水;系数越大衰减越狠)
    atten_r: float
    atten_g: float
    atten_b: float
    veil: tuple[float, float, float]  # 背散射水色(归一化 RGB)
    veil_strength: float  # 雾幕最大混合比 [0,1]
    darken: float  # 低照:亮度最大衰减比 [0,0.6]
    depth_grad: float  # 纵向深度梯度(画面下部更"深") [0,1]


def random_water(rng: np.random.Generator) -> Water:
    """采样一种水体。蓝水/绿水两族,覆盖 UVEB 总结的主要色偏类型。"""
    if rng.uniform() < 0.5:  # 蓝水(开阔海)
        veil = (rng.uniform(0.0, 0.1), rng.uniform(0.25, 0.5), rng.uniform(0.5, 0.8))
    else:  # 绿水(近海/湖)
        veil = (rng.uniform(0.05, 0.2), rng.uniform(0.45, 0.7), rng.uniform(0.2, 0.45))
    return Water(
        atten_r=float(rng.uniform(1.2, 3.0)),
        atten_g=float(rng.uniform(0.4, 1.0)),
        atten_b=float(rng.uniform(0.2, 0.7)),
        veil=tuple(float(v) for v in veil),  # type: ignore[arg-type]
        veil_strength=float(rng.uniform(0.3, 0.7)),
        darken=float(rng.uniform(0.1, 0.5)),
        depth_grad=float(rng.uniform(0.2, 0.8)),
    )


def apply_water(arr_u8: np.ndarray, water: Water, t: float) -> np.ndarray:
    """以强度 t 施加水下退化。t=0 位级恒等;t=1 完整退化。确定性、逐参数随 t 线性。"""
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"strength t out of range: {t}")
    if t == 0.0:
        return arr_u8.copy()
    x = arr_u8.astype(np.float64) / 255.0
    h = x.shape[0]

    # 画面下部更深:深度图 d(y) ∈ [1-grad, 1],随 t 不变(几何属性)
    depth = 1.0 - water.depth_grad * (1.0 - np.linspace(0, 1, h))[:, None]

    # 通道衰减:t_c = exp(-atten_c * t * depth)
    for c, atten in enumerate((water.atten_r, water.atten_g, water.atten_b)):
        trans = np.exp(-atten * t * depth)
        x[..., c] = x[..., c] * trans + water.veil[c] * water.veil_strength * t * (1.0 - trans)

    # 低照
    x = x * (1.0 - water.darken * t)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def gen_dataset(
    out_dir: Path,
    n_images: int = 6,
    n_waters: int = 4,
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seed: int = 0,
) -> Path:
    """产出 (source=干净图, result=退化图) 数据集 + manifest.jsonl(gt_strength=退化程度)。
    评估器契约:感知质量分应随 gt_strength 单调下降(注意方向与 color 域相反)。"""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [gen_test_image(rng) for _ in range(n_images)]
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w") as f:
        for wi in range(n_waters):
            water = random_water(rng)
            for ii in range(n_images):
                for t in strengths:
                    case_id = f"water{wi}_img{ii}_t{int(t * 100):03d}"
                    cdir = out_dir / case_id
                    cdir.mkdir(exist_ok=True)
                    p_src, p_res = cdir / "source.png", cdir / "result.png"
                    Image.fromarray(imgs[ii]).save(p_src)
                    Image.fromarray(apply_water(imgs[ii], water, t)).save(p_res)
                    f.write(
                        json.dumps(
                            {
                                "case_id": case_id,
                                "source": str(p_src),
                                "result": str(p_res),
                                "gt_strength": t,
                                "group": f"water{wi}_img{ii}",
                                "water": asdict(water),
                            }
                        )
                        + "\n"
                    )
    return manifest
