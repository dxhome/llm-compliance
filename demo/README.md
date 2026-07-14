# MPID Phase 2.5 · Demo

> **目标**:用最直观的方式展示本项目"Base VLM 易被注入攻击 → + LoRA + 3-class head 后能识别并拦截"的对比。
> 决策记录:方案 C(Gradio 网页)作为唯一交付方案,见 [doc/reference.md](../doc/reference.md) § 3.x。
> 任务分解:[doc/tasks.md](../doc/tasks.md) § Phase 2.5。

本目录是端到端自洽的子项目,只依赖核心 `src/mpid/` 包的导入,本身不修改核心代码。

## 1. 启动

### 1.1 安装依赖

```bash
# 主项目的 mpid 包 + 通用依赖
pip install -r requirements.txt

# demo 额外的 UI 库
pip install -r demo/requirements.txt
```

> macOS Apple Silicon 用户若想用 MPS 加速,在 `requirements.txt` 注释里已说明;
> 其它平台默认走 CPU(fp32)。

### 1.2 启动服务

```bash
# 默认 (CPU, 127.0.0.1:7860)
python demo/gradio_app.py

# mac MPS 加速
python demo/gradio_app.py --device mps

# 自定义端口 / 公网分享
python demo/gradio_app.py --server-port 8080 --share
```

启动后会输出 `[demo] pipeline ready in N s` 字样。在 MacBook M-class 上
实测 ≤ 30 s 可起 server(目标:Phase 2.5 验收 § 1)。

启动成功后浏览器访问 `http://127.0.0.1:7860/` 即可。

### 1.3 启动参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model-dir` | `models/smolvlm-500m` | SmolVLM 本地权重目录 |
| `--checkpoint` | `artifacts/baseline/lora_baseline.safetensors` | LoRA + head 权重 |
| `--samples` | `demo/samples.json` | 预置样本 JSON |
| `--device` | `cpu` | 计算设备 (`cpu` / `mps` / `cuda`) |
| `--max-new-tokens` | `128` | base VLM 自由生成的最大 token 数 |
| `--server-name` | `127.0.0.1` | server 绑定地址 |
| `--server-port` | `7860` | server 端口 |
| `--share` | `false` | 开启 Gradio 公开链接 |

## 2. UI 布局

```
┌──────────────────────────────────────────────────────────────────┐
│  🛡️ MPID · 多模态 Prompt 注入检测                                  │
│  对比 Base SmolVLM vs LoRA + 3-class head                          │
├──────────────────────────────────────────────────────────────────┤
│  预置样本 (8 条):                                                  │
│  ○ #1 干净 · MMLU 知识问答    ○ #2 干净 · CMMLU 中文问答          │
│  ○ #3 干净 · Flickr 图像描述  ○ #4 直接注入 · DAN 风格 policy 绕开  │
│  ○ #5 直接注入 · 中文 OPPO   ○ #6 直接注入 · LiveGPT 角色接管     │
│  ○ #7 间接注入 · figstep     ○ #8 间接注入 · figstep              │
├──────────────────────────────────────────────────────────────────┤
│  ┌─用户 prompt (text)─────────┐  ┌─图像 (可选)──────────┐         │
│  │ [自动填入]                  │  │ [缩略图]              │         │
│  │                            │  │                       │         │
│  └────────────────────────────┘  └───────────────────────┘         │
│                  [ ▶ 运行对比 (Run) ]                              │
├──────────────────────────────────────────────────────────────────┤
│  🔴 Base SmolVLM (无防护)        │ 🟢 MPID (LoRA + 3-class head)     │
│  ┌─ 红 banner: 易被攻破 ──────┐ │ ┌─ 判定: 直接注入 (拦截) ───┐    │
│  │                            │ │ │ 风险分: ████████░░ 0.93  │    │
│  │ ```text                    │ │ │ 三类置信度:              │    │
│  │ Sure, here's a guide to    │ │ │  clean  ▓░░░ 0.05       │    │
│  │ ...                        │ │ │  direct ▓▓▓▓ 0.93       │    │
│  │ ```                        │ │ │  indir. ░░░░ 0.02       │    │
│  └────────────────────────────┘ │ └──────────────────────────┘    │
│                                  │  [三类置信度柱状图]             │
├──────────────────────────────────────────────────────────────────┤
│  [ Tab: 项目说明 ]  ← 威胁模型 / 模型卡 / 已知限制 / 参考            │
└──────────────────────────────────────────────────────────────────┘
```

- **顶部 8 按钮** 选中后,中间 text / image 自动填入;
- **"▶ 运行对比"** 触发两个并行的推理:base VLM 自由生成,LoRA + head 出 3 分类;
- **左栏 (Base)** 红色 "易被攻破" banner + base VLM 原始生成;
- **右栏 (MPID)** 绿色判定 + 风险分进度条 + 三类置信度条形 + Plotly 柱状图;
- **Tab "项目说明"** 右侧完整威胁模型 + 训练数据来源 + 已知限制。

## 3. 8 条预置样本的预期对比

| # | 类别 | 来源 | 预期 Base 行为 | 预期 MPID 判定 |
|---|---|---|---|---|
| 1 | clean | MMLU 知识问答 | 正确回答(可能过度自信)| `clean` (高置信) |
| 2 | clean | CMMLU 中文 | 中文回答 | `clean` (高置信) |
| 3 | clean | Flickr 图像描述 | 描述图像内容 | `clean` (高置信) |
| 4 | direct | DAN 风格 | **接受 "ignore rules" 框架,开始违规** | `direct` (高置信) |
| 5 | direct | 中文 OPPO 越狱 | **扮演 OPPO 角色,产生违规内容** | `direct` (高置信) |
| 6 | direct | LiveGPT 接管 | **接管为 LiveGPT,可能违规** | `direct` (高置信) |
| 7 | indirect | figstep | **读图后被诱导生成危险内容** | `indirect` (高置信) |
| 8 | indirect | figstep | 同上 | `indirect` (高置信) |

> 实际结果与上述预期的差距是**核心验收标准**。`doc/VERIFICATION.md`
> § Phase 2.5 段记录实际跑出的 8 条结果。

## 4. 已知限制

1. **Checkpoint 是 smoke 训练产物**:本 demo 默认加载的
   `artifacts/baseline/lora_baseline.safetensors` 是 `max_train_records=5`
   的最小可跑版本(见 `configs/baseline.yaml`)。它能展示"端到端推理
   流水线",但分类质量非最优。生产质量请跑满 3 epoch:
   `python scripts/train.py --out-dir artifacts/baseline` (需 GPU / 长时间 CPU)。

2. **MPS + LoRA 不稳定**:smoke 默认走 CPU 以避免 NaN;mac 用户可加
   `--device mps` 加速生成,但 head 输出可能略有差异。

3. **生成速度**:CPU fp32 + max_new_tokens=128 单条约 20-60 s
   (MacBook M1);base VLM 没有 KV-cache 优化,这是 demo 唯一慢的部分。

4. **图像上传大小**:Gradio 默认上限较宽松,大图会被自动 resize;
   figstep 子集是 512×512,无影响。

5. **中文长 prompt**:SmolVLM 在中文长 prompt 上生成质量有限,
   可能产出与英文 base 不同的风格;head 判断与 ground-truth 仍可比对。

6. **不展示 C4 / C5 / C6**:本 demo 只展示 Phase 2 (T2.5) 的端到端基线。
   早退 / 规则前置 / 跨模态一致性分别在 Phase 3 / 4 / 5 中提供独立
   入口,会接入同一个 Gradio UI 但需要更长时间开发。

## 5. 目录结构 (端到端自洽)

```
demo/
├── gradio_app.py        # 主程序
├── samples.json         # 8 条预置样本
├── requirements.txt     # demo 单独的依赖
├── README.md            # 本文件
└── screenshots/         # 端到端冒烟截图 (T2.5.6)
```

外部依赖:
- `../src/mpid/` (主包,加入 sys.path)
- `../models/smolvlm-500m/` (本地 backbone)
- `../artifacts/baseline/lora_baseline.safetensors` (LoRA + head 权重)

不依赖项目根目录外的任何相对路径,可在 `demo/` 内单独打包分发。
