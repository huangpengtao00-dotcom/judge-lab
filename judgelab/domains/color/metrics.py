"""传统色差指标复现:ΔE76 / ΔE2000(CIEDE2000)。本仓的"被证伪对象"之一。

红线 2:公式从原论文原样抄,不自写口径。
- ΔE2000 严格按 Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference Formula:
  Implementation Notes, Supplementary Test Data, and Mathematical Observations",
  Color Research & Application 30(1):21-30 的分步实现(Eq. 2-22),含全部不连续点修正。
- 官方 34 组测试向量(Sharma 主页 ciede2000testdata.txt,经 Web Archive 取原件)
  全部进 tests/test_metrics.py,容差 1e-4。

Lab 约定:与 judgelab.core.evidence.srgb_to_lab 一致(CIE D65,L∈[0,100],a/b 原生单位)。
本模块只吃 Lab 三元组,不做 RGB 转换 —— 转换职责在 evidence.srgb_to_lab,不重复实现。
"""

from __future__ import annotations

import numpy as np

__all__ = ["delta_e76", "delta_e2000"]


def _split_lab(lab1, lab2) -> tuple[np.ndarray, ...]:
    """校验并拆开两组 Lab。形状必须 (..., 3) 且可广播,否则显式抛(红线1)。"""
    a1 = np.asarray(lab1, dtype=np.float64)
    a2 = np.asarray(lab2, dtype=np.float64)
    if a1.shape[-1:] != (3,) or a2.shape[-1:] != (3,):
        raise ValueError(f"Lab inputs must have trailing dim 3, got {a1.shape} vs {a2.shape}")
    return a1[..., 0], a1[..., 1], a1[..., 2], a2[..., 0], a2[..., 1], a2[..., 2]


def delta_e76(lab1, lab2) -> np.ndarray | float:
    """CIE76 色差:Lab 空间欧氏距离。

    出处:CIE 1976 (L*a*b*) 色差公式,ΔE*ab = sqrt(ΔL² + Δa² + Δb²)
    (见 Sharma, Wu & Dalal 2005 引言部分对 ΔE*ab 的记法)。
    已知批评:Lab 感知均匀性差,尤其蓝区与高饱和区,同一 ΔE76 数值在不同色区
    对应的感知差异可差数倍 —— 这正是 CIEDE2000 出现的原因。
    """
    L1, a1, b1, L2, a2, b2 = _split_lab(lab1, lab2)
    out = np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)
    return float(out) if out.ndim == 0 else out


def delta_e2000(lab1, lab2, k_l: float = 1.0, k_c: float = 1.0, k_h: float = 1.0):
    """CIEDE2000 色差,严格按 Sharma, Wu & Dalal 2005 的分步公式实现。

    出处:Sharma, Wu & Dalal (2005), Color Res. & Appl. 30(1):21-30。
    步骤对应论文 Eq. (2)-(22):
      C'/h' 变换 Eq. 2-7;Δ 项 Eq. 8-11(含 Δh' 的 ±360° 修正与 C1'C2'=0 特例);
      均值项 Eq. 12-14(含 h̄' 的四分支,C1'C2'=0 时取 h1'+h2',见论文注记);
      T Eq. 15;Δθ Eq. 16;R_C Eq. 17;S_L/S_C/S_H Eq. 18-20;R_T Eq. 21;ΔE00 Eq. 22。
    参数化因子 kL=kC=kH=1(论文测试向量的取值)。
    已知批评:公式分段不连续(hue 均值分支),且作为逐像素指标忽略空间上下文,
    对图像整体观感(如对比度/结构变化)不敏感 —— 只度量"点色差"。

    支持广播:标量三元组或 (..., 3) 数组均可;返回标量或数组。
    """
    L1, a1, b1, L2, a2, b2 = _split_lab(lab1, lab2)

    # --- Eq. 2-7: a' 缩放与 C', h' ---
    c1_ab = np.hypot(a1, b1)
    c2_ab = np.hypot(a2, b2)
    c_bar_ab = (c1_ab + c2_ab) / 2.0
    pow25_7 = 25.0**7
    g = 0.5 * (1.0 - np.sqrt(c_bar_ab**7 / (c_bar_ab**7 + pow25_7)))  # Eq. 4
    a1p = (1.0 + g) * a1  # Eq. 5
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)  # Eq. 6
    c2p = np.hypot(a2p, b2)
    # Eq. 7:h' = atan2(b, a') 折到 [0, 360);a'=b=0 时定义 h'=0(atan2(0,0)=0 已满足)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    # --- Eq. 8-11: ΔL', ΔC', Δh', ΔH' ---
    dLp = L2 - L1  # Eq. 8
    dCp = c2p - c1p  # Eq. 9
    dh = h2p - h1p
    zero_chroma = (c1p * c2p) == 0.0
    dhp = np.where(
        zero_chroma,
        0.0,
        np.where(np.abs(dh) <= 180.0, dh, np.where(dh > 180.0, dh - 360.0, dh + 360.0)),
    )  # Eq. 10
    dHp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp) / 2.0)  # Eq. 11

    # --- Eq. 12-14: 均值项 ---
    l_bar = (L1 + L2) / 2.0  # Eq. 12
    c_bar_p = (c1p + c2p) / 2.0  # Eq. 13
    hsum = h1p + h2p
    habs = np.abs(h1p - h2p)
    # Eq. 14:C1'C2'=0 → h̄'=h1'+h2'(论文明示取和,不取半)
    h_bar_p = np.where(
        zero_chroma,
        hsum,
        np.where(
            habs <= 180.0,
            hsum / 2.0,
            np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0),
        ),
    )

    # --- Eq. 15-21: 加权与旋转项 ---
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )  # Eq. 15
    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))  # Eq. 16
    r_c = 2.0 * np.sqrt(c_bar_p**7 / (c_bar_p**7 + pow25_7))  # Eq. 17
    s_l = 1.0 + (0.015 * (l_bar - 50.0) ** 2) / np.sqrt(20.0 + (l_bar - 50.0) ** 2)  # Eq. 18
    s_c = 1.0 + 0.045 * c_bar_p  # Eq. 19
    s_h = 1.0 + 0.015 * c_bar_p * t  # Eq. 20
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c  # Eq. 21

    # --- Eq. 22: 总色差 ---
    tl = dLp / (k_l * s_l)
    tc = dCp / (k_c * s_c)
    th = dHp / (k_h * s_h)
    out = np.sqrt(tl**2 + tc**2 + th**2 + r_t * tc * th)
    return float(out) if out.ndim == 0 else out
