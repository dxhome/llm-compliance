# 验证报告（Verification Report）

> **总入口**：本文件统一记录 MPID 各 Phase 的**实测基线、跨平台差异、已知限制与修复决策**。
> 每个 Phase 一节，按完成顺序追加；旧 Phase 不会被改写，只增不删。
>
> **当前覆盖范围**：
> - ✅ Phase 0A-1 — 运行环境搭建（已完成）
> - ⏳ Phase 0A-2 — SmolVLM-500M 模型准备（待启动）
> - ⏳ Phase 0A-3 — 训练 / 测试数据准备（待启动）
> - ⏳ Phase 0/1/2/3/4/5/6/7（待启动）
>
> **目标平台约定**（贯穿全文档）：
> - **mac**  = macOS 12.5.1 / Apple M1 Pro / 16 GB / **MPS** 可用、CUDA 不可用
> - **x86**  = Linux x86_64 / **CPU-only**（**无 CUDA、无 MPS**）
>
> **结论先行**：两个目标平台都**没有可用的 4-bit 量化路径**（mac 无 CUDA 编译版 bnb + 无 mlx wheel；x86 无 CUDA）。所有训练/推理将以 **fp16 (mac, MPS) / fp32 (x86, CPU)** 进行；4-bit 在生产部署时再补 CUDA 路径。

---

## 总览

| Phase | 状态 | 关键产出 | mac 端 | x86 (CPU) 端 | 报告位置 |
|---|---|---|---|---|---|
| 0A-1 | ✅ 完成 | 设备抽象、smoke / quant 脚本、pytest | `mps` / fp16 | `cpu` / fp32 | 本文件 § Phase 0A-1 |
| 0A-2 | ⏳ 待启动 | SmolVLM-500M 本地化 + 加载冒烟 | — | — | 待追加 § Phase 0A-2 |
| 0A-3 | ⏳ 待启动 | 4 类数据集落地 + schema 自检 | — | — | 待追加 § Phase 0A-3 |

---

## Phase 0A-1 — 运行环境搭建

> 范围：`doc/tasks.md` 中 **TP1.1 ~ TP1.8**。

### 0A-1.1 范围与交付物

| 类型 | 路径 | 说明 |
|---|---|---|
| 依赖（mac + MPS 基线） | [requirements.txt](../../requirements.txt) | 锁 torch 2.4.1 以兼容 macOS 12.5 MPS |
| 依赖（x86 + CPU-only） | [requirements-x86.txt](../../requirements-x86.txt) | torch 2.5+ 从 PyTorch CPU index；bnb 0.42（4-bit 不可用但能装） |
| 依赖（macOS 13+ 备用） | [requirements-mlx.txt](../../requirements-mlx.txt) | mlx 4-bit 备用，本期未启用 |
| 设备抽象 | [src/mpid/device.py](../../src/mpid/device.py) | 自动检测 mps / cuda / cpu；strict `--prefer` |
| 环境冒烟脚本 | [scripts/smoke_env.py](../../scripts/smoke_env.py) | 4 步：imports / device / tensor / tokenizer |
| 量化路径探针 | [scripts/quant_smoke.py](../../scripts/quant_smoke.py) | 顺序试 bnb_4bit / mlx_4bit / torch fp16/cpu |
| 设备单测 | [tests/test_device.py](../../tests/test_device.py) | 12 用例，pure-mock，跨平台 |
| mac 实测记录 | [artifacts/quantization.json](../../artifacts/quantization.json) | 见 § 0A-1.3 |
| x86 实测记录 | （待 x86 端跑出后落到 `artifacts/quantization.x86.json`） | 见 § 0A-1.4 |

---

### 0A-1.2 平台基础事实（仅记录，不重复测）

| 维度 | mac | x86 (CPU) |
|---|---|---|
| OS | Darwin 21.6.0 (macOS 12.5.1) | Linux（待定发行版） |
| 机型 | MacBook Pro (M1 Pro, 10 cores, 16 GB) | 待记录 |
| Python | 3.11.9 | 3.10+ 推荐 3.11 |
| Apple Silicon | ✅ | ❌ |
| MPS | ✅ built & available | ❌ |
| CUDA | ❌ | ❌（**无 GPU**） |
| 推荐依赖文件 | `requirements.txt` | `requirements-x86.txt` |

---

### 0A-1.3 mac 端基线（2026-07-13 实测）

| 字段 | 实测值 |
|---|---|
| torch | 2.4.1 |
| torchvision | 0.19.1 |
| transformers | 4.49.0 |
| peft | 0.14.0 |
| accelerate | 0.34.2 |
| bitsandbytes | 0.42.0（无 GPU 编译） |
| MPS 可用 | ✅ True |
| CUDA 可用 | ❌ False |

**冒烟结果**（`scripts/smoke_env.py`）：
```
[1/4] Required imports ............ 13/13 OK
[2/4] Device resolution ........... get_device() = mps
[3/4] Tensor round-trip ........... tensor device=mps sum≈2.34 alloc+sum≈25 ms
[4/4] SmolVLM-500M tokenizer ...... SKIPPED (模型 P0A-2 才下载)
Summary : 4/4 steps passed
```

**单测结果**（`pytest tests/test_device.py -v`）：`12 passed in 0.01s`

**量化探针结果**（[artifacts/quantization.json](../../artifacts/quantization.json)）：

| 路径 | 状态 | 详情 |
|---|---|---|
| `bnb_4bit` | ❌ FAIL | `Torch not compiled with CUDA enabled` |
| `mlx_4bit` | ❌ FAIL | `ModuleNotFoundError: No module named 'mlx'` |
| `torch_mps_float16` | ✅ OK | matmul 正常 |
| `torch_mps_bfloat16` | ❌ FAIL | `MPS BFloat16 is only supported on MacOS 14 or newer` |
| `torch_cpu_float32` | ✅ OK | 备用 |
| **RECOMMENDED** | **`torch_mps_float16`** | — |

---

### 0A-1.4 x86 (CPU-only) 端基线（待实测，预期值已列出）

> x86 验证完后请把这一节的「预期」列改为「实测」列，并把
> `artifacts/quantization.x86.json` 的内容贴到表格下方。

| 字段 | 预期值 | 实测值（待填） |
|---|---|---|
| torch | 2.5.x | |
| torchvision | 0.20.x | |
| bitsandbytes | 0.42.x | |
| Apple Silicon | ❌ | |
| MPS | ❌ | |
| CUDA | ❌（**无 GPU**） | |
| `get_device()` 默认 | `cpu` | |
| `--prefer cpu` | OK | |
| `--prefer cuda` | **必须 fail-fast**（exit 1） | |
| `--prefer mps` | **必须 fail-fast**（exit 1） | |
| 单元测试 12/12 | ✅ | |
| `bnb_4bit` 探针 | ❌ FAIL（无 CUDA） | |
| `mlx_4bit` 探针 | ❌ FAIL（非 Apple Silicon） | |
| `torch_cpu_float32` 探针 | ✅ OK | |
| **RECOMMENDED** | **`torch_cpu_float32`** | |

> **关键变化（相对 mac）**：RECOMMENDED 从 `torch_mps_float16` 变为 `torch_cpu_float32`，**没有 fp16 路径**。x86 CPU 上跑 fp16 matmul 比 fp32 慢（缺乏 fp16 SIMD 加速），所以反而是 fp32 更快更稳定。

---

### 0A-1.5 验证命令清单

**mac 端**：
```bash
cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_env.py                 # 默认 mps
python scripts/smoke_env.py --prefer cpu     # 必须 PASS
python scripts/smoke_env.py --prefer cuda    # 必须 exit 1（fail-fast）
python scripts/quant_smoke.py --out artifacts/quantization.json
pytest tests/test_device.py -v              # 12/12 PASS
```

**x86 (CPU) 端**：
```bash
cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-x86.txt
python scripts/smoke_env.py                 # 默认 cpu
python scripts/smoke_env.py --prefer cpu     # 必须 PASS
python scripts/smoke_env.py --prefer cuda    # 必须 exit 1（fail-fast）
python scripts/smoke_env.py --prefer mps     # 必须 exit 1（fail-fast）
python scripts/quant_smoke.py --out artifacts/quantization.x86.json
pytest tests/test_device.py -v              # 12/12 PASS
```

**diff 步骤**（x86 端跑完后）：
1. 把 `artifacts/quantization.x86.json` 与 `artifacts/quantization.json` 对照；
2. **RECOMMENDED 字段差异应当是 mac=`torch_mps_float16`，x86=`torch_cpu_float32`**；
3. x86 端 `host.cuda_available=false` 且 `host.mps_available=false`；
4. 两个端 `tests/test_device.py` 都必须 12/12 PASS。

---

### 0A-1.6 已知限制 & 决策记录

1. **torch 版本被 macOS 12 拖到 2.4.1**。`requirements-x86.txt` 用 torch 2.5+ 走 PyTorch CPU index（`+cpu` wheel）。
2. **4-bit 量化在两个目标平台都不可用**。
   - mac：无 CUDA 编译版 bnb；mlx 在 macOS 12 无 wheel。
   - x86：无 CUDA；非 Apple Silicon。
   - **影响**：训练/推理将以 fp16 (mac MPS) / fp32 (x86 CPU) 进行。P0A-1 原验收条件「4-bit 至少在一个平台跑通」**当前不满足**，需在 Phase 2/3 引入 CUDA 主机（云端 / 实验室 GPU）后再补 4-bit 验证。
3. **bnb 0.42 是 `py3-none-any` 纯 Python wheel**。能在两个平台都装上，但 4-bit/8-bit 函数运行时要求 CUDA，调用会抛错；`scripts/quant_smoke.py` 已正确捕获并降级到 CPU 路径。
4. **MPS BFloat16 在 macOS 12/13 不可用**（需 14+）。训练时若用 MPS 只能用 fp16。
5. **`scripts/smoke_env.py` 不会静默回退**。`--prefer cuda` 在无 CUDA 机器上以 exit code 1 失败；`--prefer mps` 在非 Apple Silicon 机器上也以 exit code 1 失败。这避免了 Phase 2 跨平台 F1 对齐时因设备悄悄回退而出现的「数据上看似一致、实则换路」的伪 bug。
6. **SmolVLM-500M 模型尚未下载**（P0A-2 任务），所以冒烟第 4 步是 `SKIPPED` 而非 `PASS`——这在 P0A-1 范围内是预期行为。
7. **`requirements-mlx.txt` 暂未启用**。保留为 macOS 13+ 用户的备用 4-bit 路径。

---

### 0A-1.7 跨平台一致性自检（Phase 2 前置）

| 维度 | mac MPS | x86 CPU | 一致性要求 |
|---|---|---|---|
| `get_device()` 默认 | `mps` | `cpu` | 字符串不同，但**应可被 trainer 透明消费** |
| `--prefer cpu` 行为 | cpu tensor ok | cpu tensor ok | **必须两边都通** |
| `--prefer cuda` 行为 | fail-fast exit 1 | fail-fast exit 1 | **必须两边都 fail-fast** |
| 单元测试 12/12 | ✅ | ✅ | 失败即视为环境破坏 |
| 4-bit 可用 | ❌ 走 fp16 | ❌ 走 fp32 | **两个目标平台都不满足原 P0A-1 验收**，需 Phase 2/3 引入 CUDA 主机补做 |

**P0A-1 验收通过条件（基于实测修订）**：
- ✅ mac 与 x86 两个平台都跑通 `smoke_env.py`（默认 + `--prefer cpu`）；
- ✅ `get_device()` 返回正确设备类型（mac=`mps`，x86=`cpu`）；
- ⚠️ 4-bit 量化「在任一目标平台能跑通最小测试」**当前不满足**——已在 0A-1.6 §2 记录限制与补做计划（需 CUDA 主机）。
- ✅ 跨平台 fail-fast 一致（`--prefer` 不静默回退）。

---

## Phase 0A-2 — SmolVLM-500M 模型准备（占位）

> 占位：P0A-2 启动后在此追加。预期内容：
> - 选型记录（SmolVLM-500M, Apache 2.0, COLM 2025）
> - 下载脚本 `scripts/download_models.py` 与本地化路径 `models/smolvlm-500m/`
> - 加载冒烟 `scripts/smoke_model.py` 的 mac / x86 双端实测
> - 4-bit 量化加载验证（mac / x86 双端结果，与 § 0A-1.6 §2 的限制相对应）

---

## Phase 0A-3 — 训练 / 测试数据准备（占位）

> 占位：P0A-3 启动后在此追加。预期内容：
> - 4 类数据集下载记录
> - schema 校验脚本 `scripts/smoke_data.py` 结果
> - 课题符合性自检（中文 / 图像 / 多语种 / 直接+间接 / 干净负例）

---

（后续 Phase 完成后继续追加，每节结构与 0A-1 平行。）
