"""从 demo_data.json 组装单文件演示页 demo.html(图片 base64 内嵌,离线可开)。
重新生成数据:见 git log 中生成 demo_data.json 的脚本;本文件只做组装,保证演示可复现。"""

import json
from pathlib import Path

HERE = Path(__file__).parent
d = json.loads((HERE / "demo_data.json").read_text())


def svg_curves() -> str:
    """两条评估器分-vs-真实强度曲线,内联 SVG,坐标脚本算。"""
    w, h, pad = 460, 260, 44
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]

    def pts(curve: dict) -> str:
        out = []
        for t in ts:
            x = pad + t * (w - 2 * pad)
            y = h - pad - (curve[str(t)] - 0.5) / 0.5 * (h - 2 * pad)
            out.append(f"{x:.1f},{y:.1f}")
        return " ".join(out)

    axis_ticks = "".join(
        f'<text x="{pad + t * (w - 2 * pad):.0f}" y="{h - pad + 18}" text-anchor="middle" class="tick">{t}</text>'
        for t in ts
    )
    y_ticks = "".join(
        f'<text x="{pad - 8}" y="{h - pad - (v - 0.5) / 0.5 * (h - 2 * pad) + 4:.0f}" text-anchor="end" class="tick">{v}</text>'
        for v in (0.5, 0.75, 1.0)
    )
    return f"""<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px">
  <line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#94a3b8"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#94a3b8"/>
  {axis_ticks}{y_ticks}
  <text x="{w / 2}" y="{h - 6}" text-anchor="middle" class="lbl">真实追色强度 t(合成时已知)</text>
  <text x="14" y="{h / 2}" text-anchor="middle" class="lbl" transform="rotate(-90 14 {h / 2})">评估器打分(均值)</text>
  <polyline points="{pts(d['curve_selfref'])}" fill="none" stroke="#059669" stroke-width="3"/>
  <polyline points="{pts(d['curve_real'])}" fill="none" stroke="#dc2626" stroke-width="3" stroke-dasharray="7 4"/>
  <text x="{w - pad}" y="{pad - 4}" text-anchor="end" class="leg" fill="#059669">伪GT设定 ρ={d['rho_selfref']}</text>
  <text x="{w - pad}" y="{pad + 16}" text-anchor="end" class="leg" fill="#dc2626">真实设定 ρ={d['rho_real']}</text>
</svg>"""


html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>追色评测失效 · 30秒演示</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;background:#f5f6f8;color:#1a2332;line-height:1.7;padding:20px 12px;max-width:860px;margin:0 auto}}
h1{{font-size:21px}} h2{{font-size:15px;margin:18px 0 8px}}
.card{{background:#fff;border:1px solid #e5e9f0;border-radius:12px;padding:18px 20px;margin:12px 0}}
.imgs{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.imgs figure{{margin:0}} .imgs img{{width:100%;border-radius:6px;display:block}}
.imgs figcaption{{font-size:12px;text-align:center;color:#5b6b83;margin-top:4px}}
.tick{{font-size:11px;fill:#5b6b83}} .lbl{{font-size:12px;fill:#1a2332}} .leg{{font-size:13px;font-weight:700}}
.pt{{background:#eff6ff;border-radius:8px;padding:10px 14px;font-size:14px;margin:8px 0}}
code{{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:12.5px}}
.big{{font-size:26px;font-weight:800}} .g{{color:#059669}} .r{{color:#dc2626}}
</style></head><body>
<h1>「追色追准没有」现在没有可靠的自动评测 —— 30 秒实验</h1>

<div class="card">
<h2>① 任务长这样:参考图和原图内容不同,只追"色调"</h2>
<div class="imgs">
<figure><img src="data:image/png;base64,{d['source']}"><figcaption>原图(待调色)</figcaption></figure>
<figure><img src="data:image/png;base64,{d['reference']}"><figcaption>参考图(目标色调,<b>内容不同</b>)</figcaption></figure>
<figure><img src="data:image/png;base64,{d['result_t50']}"><figcaption>追了一半(t=0.5)</figcaption></figure>
<figure><img src="data:image/png;base64,{d['result_t100']}"><figcaption>追满(t=1.0)</figcaption></figure>
</div>
<div class="pt">合成数据里"追到几成"(t)是<b>已知答案</b>——好的评测指标,打分必须随 t 单调上升。</div>
</div>

<div class="card">
<h2>② 同一个直方图指标,两种设定,判若两人</h2>
{svg_curves()}
<div class="pt"><span class="big g">ρ={d['rho_selfref']}</span>&nbsp; 伪GT设定(参考图=同一张原图调好色):近乎完美 —— <b>现有 benchmark 都这么造题</b><br>
<span class="big r">ρ={d['rho_real']}</span>&nbsp; 真实设定(参考图内容不同):曲线躺平,量程塌掉 —— <b>真实业务就是这个设定,靠人眼</b></div>
</div>

<div class="card">
<h2>③ 那 VLM 直接当评估器行不行?(20 条实测)</h2>
<div class="pt">zero-shot VLM 评估器在同一真实设定下:<span class="big g">ρ=0.99</span>(20/20 解析成功,约 12s/条)—— VLM 是对的底座。<br>
但文献已证明<b>未训练的 VLM 评估器会被刷分击穿</b>(JarvisEvo 消融:静态大模型当 reward,训到后期自评分升、真实质量降),理由也无证据约束。</div>
</div>

<div class="card">
<h2>④ 复现与结论</h2>
<p><code>cd ~/research/judge-lab && uv run pytest -q</code>(含本对比的断言)· n=150×2 设定 + 20 条 VLM 冒烟,seed=0,零人工标注 —— 数据由参数化 Look 程序化合成,答案天然已知。</p>
<div class="pt">三步链条:传统指标真实设定下崩(0.58)→ zero-shot VLM 能扛(0.99)但可被刷、无证据 → 要做的是<b>训练过的、吃三元组、每一分都带证据引用</b>的评估器。</div>
</div>
</body></html>"""

(HERE / "demo.html").write_text(html)
print(f"written: {HERE / 'demo.html'} ({len(html) // 1024} KB)")
