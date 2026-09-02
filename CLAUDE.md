# judge-lab

评测评估器研究仓。方案与背景见 `~/Obsidian/申请/Jarvis学习材料/16-代码架构起步-judge-lab.md`。

## 设计红线(违反任何一条的 PR 不收)

1. **不静默降级**:证据取不到、API 失败、schema 不过 → 显式 status(`judge_input_failure` / `judge_parse_failure` / `abstain`),绝不返回空 verdict 或默认分当正常数据。
2. **判据从被测对象原样抄**:复现传统指标(ΔE/UIQM/直方图)以原论文公式为准并附测试样例,不自创口径。
3. **批量 LLM 先小样本**:api judge 跑真模型前,先 ≤20 条估成本与解析成功率,确认后再放量。
4. **证据落库 + 离线重判**:每次判卷的证据包版本、原始输出全部进 SQLite;换 prompt/模型可离线重判,不重烧 token。
5. **公司代码零引用**:任何 Meshy 内部实现只作脑内启发,一行不进本仓;同类逻辑照开源实现(VisualQuality-R1 / Flow-GRPO / meval3d 开源后)重写。
6. **证据字节版本铁律**:评估器所见字节有任何变化必须 bump `EVIDENCE_PACK_VERSION`,verdict 与版本绑定。
7. **确定性**:所有合成与采样带显式 seed;同输入必须位级同输出。
