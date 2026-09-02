"""证据包:评估器所见的一切。评估器只准基于证据包判分,evidence_refs 必须指向这里的 id。

红线 6:改动任何进入评估器视野的字节(直方图 bin 数、Lab 精度、图片编码参数)都要 bump 版本。
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import JudgeInput

EVIDENCE_PACK_VERSION = "ev1"

_HIST_BINS = 32
_THUMB_MAX = 512  # 评估器看到的图统一缩到长边 ≤512,防止超大图撑爆上下文


@dataclass
class EvidenceItem:
    id: str
    kind: str  # "image_b64" | "lab_stats" | "hist_lab"
    payload: object


@dataclass
class EvidencePack:
    case_id: str
    version: str
    items: dict[str, EvidenceItem] = field(default_factory=dict)

    def add(self, item: EvidenceItem) -> None:
        if item.id in self.items:
            raise ValueError(f"duplicate evidence id: {item.id}")
        self.items[item.id] = item

    def ids(self) -> set[str]:
        return set(self.items)


class EvidenceFailure(Exception):
    """证据构建失败。调用方必须把它转成 status=judge_input_failure,不许吞。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------- 颜色数学(sRGB -> CIELAB,D65) ----------

def srgb_to_lab(arr_u8: np.ndarray) -> np.ndarray:
    """arr_u8: HxWx3 uint8 sRGB -> HxWx3 float32 Lab。公式按 CIE 标准,勿改精度(红线6)。"""
    s = arr_u8.astype(np.float64) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_hist(lab: np.ndarray) -> np.ndarray:
    """L/a/b 三通道各 _HIST_BINS bin 的归一化直方图,拼成 (3, bins)。"""
    ranges = [(0.0, 100.0), (-110.0, 110.0), (-110.0, 110.0)]
    out = np.zeros((3, _HIST_BINS), dtype=np.float64)
    for c, (lo, hi) in enumerate(ranges):
        h, _ = np.histogram(lab[..., c], bins=_HIST_BINS, range=(lo, hi))
        total = h.sum()
        if total == 0:
            raise EvidenceFailure(f"empty histogram on channel {c}")
        out[c] = h / total
    return out


def _load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise EvidenceFailure(f"image missing: {path}")
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:  # 显式转失败,不吞(红线1)
        raise EvidenceFailure(f"unreadable image {path}: {e}") from e
    img.thumbnail((_THUMB_MAX, _THUMB_MAX))
    return np.asarray(img)


def _b64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def content_hash(inp: JudgeInput) -> str:
    """case 内容指纹:任一输入图字节变化 → 指纹变化 → runner 视为新 case。"""
    h = hashlib.sha256()
    for role in ("source", "reference", "result"):
        p: Path | None = getattr(inp, role)
        h.update(role.encode())
        h.update(p.read_bytes() if p and p.exists() else b"<absent>")
    return h.hexdigest()[:16]


def build_evidence(inp: JudgeInput, required: tuple[str, ...]) -> EvidencePack:
    """构建证据包。required 里的角色缺失/坏图 → EvidenceFailure(绝不带病判卷)。"""
    pack = EvidencePack(case_id=inp.case_id, version=EVIDENCE_PACK_VERSION)
    for role in ("source", "reference", "result"):
        p: Path | None = getattr(inp, role)
        if p is None:
            if role in required:
                raise EvidenceFailure(f"required input absent: {role}")
            continue
        arr = _load_image(p)
        lab = srgb_to_lab(arr)
        pack.add(EvidenceItem(f"img:{role}", "image_b64", _b64_png(arr)))
        pack.add(
            EvidenceItem(
                f"lab_stats:{role}",
                "lab_stats",
                {
                    "mean": [float(x) for x in lab.reshape(-1, 3).mean(0)],
                    "std": [float(x) for x in lab.reshape(-1, 3).std(0)],
                },
            )
        )
        pack.add(EvidenceItem(f"hist:{role}", "hist_lab", lab_hist(lab)))
    return pack
