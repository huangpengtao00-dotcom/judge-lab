"""传统水下画质指标复现:UCIQE / UIQM。本仓的"被证伪对象":
文献已多次指出二者会奖励过度增强(过饱和/对比拉爆),tests/test_metrics.py 里有复现测试。

红线 2:公式从原论文原样抄,不自写口径。原文未钉死的实现自由度(块大小、除零守卫、
灰度权重)在各函数注释里逐条声明,并与两份广泛引用的参考实现交叉核对:
  [R1] xueleichen/PSNR-SSIM-UCIQE-UIQM-Python nevaluate.py(Python,skimage 系)
  [R2] bilityniu/underimage-fusion-enhancement UICM/UISM/UIConM.m(MATLAB)

输入契约:HxWx3 uint8 sRGB。形状/类型不符显式抛 ValueError(红线1,不静默降级)。
Lab 转换复用 judgelab.core.evidence.srgb_to_lab(D65,L∈[0,100]),不重复实现。
"""

from __future__ import annotations

import math

import numpy as np

from ...core.evidence import srgb_to_lab

__all__ = ["uciqe", "uiqm", "uicm", "uism", "uiconm"]

# UCIQE 系数,Yang & Sowmya 2015 Table I(MOS 回归所得原值)
_UCIQE_C1, _UCIQE_C2, _UCIQE_C3 = 0.4680, 0.2745, 0.2576
# UIQM 系数,Panetta, Gao & Agaian 2016(线性组合原值)
_UIQM_C1, _UIQM_C2, _UIQM_C3 = 0.0282, 0.2953, 3.5753
# PLIP 参数 γ=k=1026(Panetta 系 PLIP 框架取值,[R1] 同)
_PLIP_GAMMA = 1026.0
# EME / logAMEE 分块边长。原论文只写"划分为 k1×k2 块"未钉死边长;
# 取 [R1] 的 8×8(注:[R2] 用 5×5,块大小会平移绝对分值,不改本仓契约测试的排序结论)。
_BLOCK = 8


def _require_rgb_u8(arr: np.ndarray, who: str) -> None:
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{who}: expect HxWx3 array, got {getattr(arr, 'shape', type(arr))}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{who}: expect uint8 sRGB, got dtype {arr.dtype}")
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        raise ValueError(f"{who}: image too small for 3x3 Sobel / block stats: {arr.shape}")


# ---------------------------------------------------------------- UCIQE


def uciqe(arr_u8: np.ndarray) -> float:
    """UCIQE = c1·σ_c + c2·con_l + c3·μ_s(CIELab 域)。

    出处:Yang & Sowmya (2015), "An Underwater Color Image Quality Evaluation
    Metric", IEEE Trans. Image Processing 24(12):6062-6071。总公式 Eq. (14),
    系数 c1=0.4680, c2=0.2745, c3=0.2576 为论文 MOS 回归原值(Table I)。
    各项定义(论文 Sec. III-B,与 [R1] 逐行核对一致):
      σ_c   = 色度标准差,色度 C = sqrt(a² + b²);
      con_l = 亮度对比 = L 通道最亮 1% 像素均值 − 最暗 1% 像素均值;
      μ_s   = 饱和度均值,逐像素 s = C / L(CIELab 定义)。
    实现声明(原文未钉死处):
      - Lab 用本仓 srgb_to_lab(D65,L∈[0,100],a/b 原生单位),不归一化 —— 与
        [R1](skimage rgb2lab)同一量纲;绝对分值依赖该量纲,跨实现比较只看排序。
      - 1% 像素数 = max(1, round(0.01·N));
      - L=0(纯黑)或 C=0 的像素饱和度记 0(C/L 在 L=0 处未定义,[R1] 同)。
    已知批评:σ_c 与 μ_s 都随饱和度单调上涨,con_l 随对比拉爆上涨 —— 对过度增强
    (过饱和、对比裁剪)给高分,与人眼评价背离(如 Li et al. 2020 UIEB 基准的讨论)。
    """
    _require_rgb_u8(arr_u8, "uciqe")
    lab = srgb_to_lab(arr_u8).astype(np.float64)
    lum = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])

    sigma_c = float(chroma.std())  # 总体标准差(σ = sqrt(E[(C-μ_C)²]))

    n = lum.size
    k = max(1, int(round(0.01 * n)))
    lum_sorted = np.sort(lum, axis=None)
    con_l = float(lum_sorted[-k:].mean() - lum_sorted[:k].mean())

    sat = np.divide(chroma, lum, out=np.zeros_like(chroma), where=lum > 0)
    mu_s = float(sat.mean())

    return _UCIQE_C1 * sigma_c + _UCIQE_C2 * con_l + _UCIQE_C3 * mu_s


# ---------------------------------------------------------------- UIQM 子项


def _trimmed_mean_var(values: np.ndarray, alpha_l: float = 0.1, alpha_r: float = 0.1):
    """非对称 alpha-trimmed 均值与围绕它的二阶矩(Panetta 2016 Sec. III-A)。

    排序后去掉左 α_L、右 α_R 比例的样本(论文默认 α_L=α_R=0.1,[R1][R2] 同),
    对剩余样本取均值与方差。样本被剪空时显式抛(红线1)。
    """
    flat = np.sort(values, axis=None)
    total = flat.size
    t_l = int(alpha_l * total)
    t_r = int(alpha_r * total)
    kept = flat[t_l : total - t_r]
    if kept.size == 0:
        raise ValueError(f"alpha-trim left no samples (n={total}, aL={alpha_l}, aR={alpha_r})")
    mu = float(kept.mean())
    var = float(((kept - mu) ** 2).mean())
    return mu, var


def uicm(arr_u8: np.ndarray) -> float:
    """UICM:水下图像色彩度(对立色统计)。

    出处:Panetta, Gao & Agaian (2016), "Human-Visual-System-Inspired Underwater
    Image Quality Measures", IEEE J. Oceanic Eng. 41(3):541-551, Sec. III-A。
    对立色通道 RG = R − G,YB = (R+G)/2 − B;各通道取 α-trimmed 均值 μ 与方差 σ²
    (α_L=α_R=0.1),合成:
      UICM = −0.0268·sqrt(μ²_RG + μ²_YB) + 0.1586·sqrt(σ²_RG + σ²_YB)
    (常数为论文原值,[R1][R2] 一致)。像素量纲 0..255 浮点([R2] 同)。
    已知批评:第二项奖励对立色方差 —— 人工拉爆饱和度直接抬高 σ²,得分上涨。
    """
    _require_rgb_u8(arr_u8, "uicm")
    x = arr_u8.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mu_rg, var_rg = _trimmed_mean_var(r - g)
    mu_yb, var_yb = _trimmed_mean_var((r + g) / 2.0 - b)
    return -0.0268 * math.hypot(mu_rg, mu_yb) + 0.1586 * math.sqrt(var_rg + var_yb)


def _sobel_mag(ch: np.ndarray) -> np.ndarray:
    """3x3 Sobel 梯度幅值 sqrt(gx²+gy²),边缘 replicate 填充。纯 numpy 实现
    (约束:不引 opencv/scikit-image)。核为教科书 Sobel:
      Gx = [[-1,0,1],[-2,0,2],[-1,0,1]],Gy = Gx^T。"""
    p = np.pad(ch, 1, mode="edge")
    left = p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2]
    right = p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]
    top = p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:]
    bottom = p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]
    return np.hypot(right - left, bottom - top)


def _iter_blocks(ch: np.ndarray, block: int):
    """按 block×block 分块遍历(末行/末列不足一块的余块也算一块,[R1] 同)。"""
    h, w = ch.shape
    k1 = math.ceil(h / block)
    k2 = math.ceil(w / block)
    for i in range(k1):
        for j in range(k2):
            yield ch[i * block : (i + 1) * block, j * block : (j + 1) * block]
    return


def _eme(ch: np.ndarray, block: int = _BLOCK) -> float:
    """EME(增强度量):EME = (2/(k1·k2))·ΣΣ ln(I_max / I_min),按块统计。

    出处:Panetta 2016 Sec. III-B(源自 Agaian 系 EME 度量);自然对数([R1][R2] 同)。
    实现声明:I_min=0 或 I_max=0 的块(除零/对数未定义,原文未处理)贡献记 0,
    与 [R2] 一致([R1] 用 min 钳到 1 的变体;两者只平移绝对分值)。
    """
    h, w = ch.shape
    k1 = math.ceil(h / block)
    k2 = math.ceil(w / block)
    total = 0.0
    for blk in _iter_blocks(ch, block):
        mn = float(blk.min())
        mx = float(blk.max())
        if mn > 0.0 and mx > 0.0:
            total += math.log(mx / mn)
    return 2.0 / (k1 * k2) * total


def uism(arr_u8: np.ndarray) -> float:
    """UISM:水下图像清晰度(Sobel 边缘 + 分块 EME)。

    出处:Panetta, Gao & Agaian (2016), Sec. III-B。每个 RGB 通道:
      1) Sobel 边缘图与原通道逐像素相乘得"灰度边缘图"(论文原话:edge map
         multiplied with the original image);
      2) 对灰度边缘图求 EME(分块 ln(max/min),块 8×8,见 _eme 的声明);
      3) 按视觉显著性加权 UISM = Σ_c λ_c·EME_c,λ_R=0.299, λ_G=0.587, λ_B=0.114
         (论文原值,即 Rec.601 亮度权重)。
    已知批评:EME 是 max/min 比值,对比拉爆/锐化过冲会直接抬高比值 —— 奖励过度
    锐化;且对噪声敏感(噪点即"边缘")。
    """
    _require_rgb_u8(arr_u8, "uism")
    x = arr_u8.astype(np.float64)
    weights = (0.299, 0.587, 0.114)
    total = 0.0
    for c, lam in enumerate(weights):
        ch = x[..., c]
        total += lam * _eme(_sobel_mag(ch) * ch)
    return total


def _plip_sub(a: float, b: float, k: float = _PLIP_GAMMA) -> float:
    """PLIP 减法 a ⊖ b = k(a−b)/(k−b)(Panetta 系 PLIP 框架,γ=k=1026)。"""
    return k * (a - b) / (k - b)


def _plip_add(a: float, b: float, gamma: float = _PLIP_GAMMA) -> float:
    """PLIP 加法 a ⊕ b = a + b − a·b/γ。"""
    return a + b - a * b / gamma


def _plip_scalar_mult(c: float, a: float, gamma: float = _PLIP_GAMMA) -> float:
    """PLIP 标量乘 c ⊗ a = γ − γ(1 − a/γ)^c。"""
    return gamma - gamma * (1.0 - a / gamma) ** c


def uiconm(arr_u8: np.ndarray, block: int = _BLOCK) -> float:
    """UIConM:水下图像对比度,logAMEE(强度图)。

    出处:Panetta, Gao & Agaian (2016), Sec. III-C:
      logAMEE = (1/(k1·k2)) ⊗ ΣΣ⊕ (I_max ⊖ I_min)/(I_max ⊕ I_min)
                × log[(I_max ⊖ I_min)/(I_max ⊕ I_min)]
    其中 ⊖/⊕/⊗ 为 PLIP 运算(γ=1026),块内 Michelson 对比的对数熵。
    实现声明(原文未钉死处,均已与参考实现交叉核对):
      - 强度图取 Rec.601 亮度 0.299R+0.587G+0.114B(与论文 UISM 的 λ 同源;
        [R1] 用 skimage 灰度权重,[R2] 用逐通道求和 —— 公开实现在此处彼此不一致,
        这本身就是该指标可复现性差的证据);
      - 像素量纲 0..255 浮点;I_max=I_min=0 的块(⊕ 得 0,除零)贡献记 0;
      - 熵项取 −m·ln(m) ≥ 0 约定(m∈(0,1) 时 m·ln m 恒负;论文式子未写符号,
        但论文 Table 报告的 UIConM 为正值,[R2] 取 |Σ| 等价);
      - 最外层按论文用 PLIP 标量乘 ⊗ 施加 1/(k1k2)([R2] 用普通除法的变体)。
    已知批评:m·ln m 熵权在 m→1(对比拉爆)与 m→0(全平)处都趋 0,非单调,
    对"适度对比"反而可能低于对比裁剪后的图;跨实现符号/灰度约定混乱。
    """
    _require_rgb_u8(arr_u8, "uiconm")
    x = arr_u8.astype(np.float64)
    gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    h, w = gray.shape
    k1 = math.ceil(h / block)
    k2 = math.ceil(w / block)
    entropy = 0.0
    for blk in _iter_blocks(gray, block):
        mn = float(blk.min())
        mx = float(blk.max())
        bottom = _plip_add(mx, mn)
        if bottom == 0.0:  # 全黑块:Michelson 对比未定义,贡献 0
            continue
        m = _plip_sub(mx, mn) / bottom
        if m > 0.0:
            entropy += -m * math.log(m)  # 对数熵,m∈(0,1) 时非负
    return _plip_scalar_mult(1.0 / (k1 * k2), entropy)


def uiqm(arr_u8: np.ndarray) -> float:
    """UIQM = c1·UICM + c2·UISM + c3·UIConM。

    出处:Panetta, Gao & Agaian (2016), IEEE J. Oceanic Eng. 41(3):541-551,
    总公式(Sec. III 末),系数 c1=0.0282, c2=0.2953, c3=3.5753(论文原值,
    面向水下彩色图的整体取值)。
    已知批评:三个子项分别奖励饱和度方差、边缘比值、块对比 —— 过度增强
    (过饱和+过锐化+对比裁剪)可同时抬高三项;与人评相关性在增强图上显著退化
    (见 UIEB 基准与 "On the limits of perceptual quality measures for enhanced
    underwater images" 等批评文献)。
    """
    _require_rgb_u8(arr_u8, "uiqm")
    return _UIQM_C1 * uicm(arr_u8) + _UIQM_C2 * uism(arr_u8) + _UIQM_C3 * uiconm(arr_u8)
