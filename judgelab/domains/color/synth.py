"""参数化调色合成:造"追色程度已知"的三元组(可验证数据,评估器训练/评测的地基)。

原理:一个 Look(色温/色调/gamma/饱和度/对比度/lift 的参数组合)是确定性变换。
  reference = Look 以强度 1.0 施加在图 B 上(内容与 A 不同 —— 这正是现有指标全体失效的设定)
  result    = Look 以强度 t 施加在图 A 上,t ∈ [0,1] 即"追色程度"的 ground truth
评估器对 (source=A, reference, result) 打的"色调追随"分,应当与 t 单调相关 —— 这就是零标注的可验证判据。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Look:
    temp: float  # 色温偏移,[-1,1],正=暖(加 R 减 B)
    tint: float  # 色调偏移,[-1,1],正=品红
    gamma: float  # [0.6, 1.6]
    saturation: float  # [0.4, 1.8]
    contrast: float  # [0.6, 1.5]
    lift: float  # 黑位提升 [0, 0.15]


def random_look(rng: np.random.Generator) -> Look:
    return Look(
        temp=float(rng.uniform(-0.6, 0.6)),
        tint=float(rng.uniform(-0.4, 0.4)),
        gamma=float(rng.uniform(0.7, 1.4)),
        saturation=float(rng.uniform(0.5, 1.6)),
        contrast=float(rng.uniform(0.7, 1.4)),
        lift=float(rng.uniform(0.0, 0.12)),
    )


def apply_look(arr_u8: np.ndarray, look: Look, t: float) -> np.ndarray:
    """以强度 t 施加 Look。t=0 恒等(位级);t=1 完整效果;参数全部按 t 线性插值。"""
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"strength t out of range: {t}")
    if t == 0.0:
        return arr_u8.copy()
    x = arr_u8.astype(np.float64) / 255.0

    temp, tint = look.temp * t, look.tint * t
    x[..., 0] = x[..., 0] + 0.12 * temp
    x[..., 2] = x[..., 2] - 0.12 * temp
    x[..., 1] = x[..., 1] - 0.08 * tint
    x = np.clip(x, 0.0, 1.0)

    gamma = 1.0 + (look.gamma - 1.0) * t
    x = x ** gamma

    lum = x @ np.array([0.2126, 0.7152, 0.0722])
    sat = 1.0 + (look.saturation - 1.0) * t
    x = np.clip(lum[..., None] + (x - lum[..., None]) * sat, 0.0, 1.0)

    contrast = 1.0 + (look.contrast - 1.0) * t
    x = np.clip((x - 0.5) * contrast + 0.5, 0.0, 1.0)

    lift = look.lift * t
    x = x * (1.0 - lift) + lift
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------- 程序化测试图(单测/POC 用;真实数据集接入是 M1 的事) ----------

def gen_test_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """梯度底 + 随机色块:直方图足够丰富,保证 Look 效果可测。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size
    base = np.stack(
        [0.25 + 0.5 * xx, 0.25 + 0.5 * yy, 0.35 + 0.3 * (xx + yy) / 2], axis=-1
    )
    for _ in range(8):
        cx, cy = rng.uniform(0, size, 2)
        r = rng.uniform(size * 0.05, size * 0.22)
        color = rng.uniform(0.05, 0.95, 3)
        mask = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < r**2
        base[mask] = color
    return (base * 255).astype(np.uint8)


def gen_dataset(
    out_dir: Path, n_images: int = 8, n_looks: int = 5, strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0
) -> tuple[Path, Path]:
    """产出三元组数据集 + 两份 manifest。确定性:同 seed 同字节。

    manifest.jsonl          真实设定:reference 内容 ≠ source 内容(现有指标全体失效的设定)
    manifest_selfref.jsonl  自参考设定:reference = 同源满强度 target(CanonCGT 式伪 GT 回避协议)
    两份共用同一批 result;同一个评估器在两份上的单调性差,就是"内容混杂杀伤力"的直接测量。
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [gen_test_image(rng) for _ in range(n_images + 1)]  # 最后一张固定当 reference 的内容图
    manifest = out_dir / "manifest.jsonl"
    manifest_selfref = out_dir / "manifest_selfref.jsonl"
    with manifest.open("w") as f, manifest_selfref.open("w") as fs:
        for li in range(n_looks):
            look = random_look(rng)
            ref_content = imgs[-1]
            for ii in range(n_images):
                src = imgs[ii]
                for t in strengths:
                    case_id = f"look{li}_img{ii}_t{int(t * 100):03d}"
                    cdir = out_dir / case_id
                    cdir.mkdir(exist_ok=True)
                    p_src, p_ref = cdir / "source.png", cdir / "reference.png"
                    p_tgt, p_res = cdir / "target.png", cdir / "result.png"
                    Image.fromarray(src).save(p_src)
                    Image.fromarray(apply_look(ref_content, look, 1.0)).save(p_ref)
                    Image.fromarray(apply_look(src, look, 1.0)).save(p_tgt)
                    Image.fromarray(apply_look(src, look, t)).save(p_res)
                    common = {"gt_strength": t, "group": f"look{li}_img{ii}", "look": asdict(look)}
                    f.write(
                        json.dumps(
                            {"case_id": case_id, "source": str(p_src), "reference": str(p_ref), "result": str(p_res), **common}
                        )
                        + "\n"
                    )
                    fs.write(
                        json.dumps(
                            {"case_id": f"{case_id}_selfref", "source": str(p_src), "reference": str(p_tgt), "result": str(p_res), **common}
                        )
                        + "\n"
                    )
    return manifest, manifest_selfref
