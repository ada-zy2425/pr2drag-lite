# PR²-Control（实验框架/代码骨架）

这是一个**可复用、可扩展、可断点续跑**的实验框架，用于实现你 proposal 里的 `PR²-Control` 控制层，并在不同编辑器（Editor-A / Editor-B）上做 Q1–Q3 的系统实验。

本仓库刻意把“控制层/策略/阈值选择/评测/日志”做成独立模块；编辑器本体通过 `pr2/editors/` 的 Adapter 接口接入（满足 H1/H2/H3 最小 hook 集）。

> 说明：由于你后续会接入真实 diffusion/video editor，本仓库默认提供一个 `DummyEditor` 用于自测管线（不做真实编辑，但能跑通全流程、产出日志和聚合结果）。

---

## 0. 安装（vast.ai 友好）

最小依赖（可直接跑 demo / 跑任务集 / 聚合）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

可选依赖：
- Tier-0 posterior 训练/校准建议装 `scikit-learn`
- 如果你用 MLP posterior 或真实 diffusion editor，装 `torch`
- 如果需要读视频/保存视频，装 `opencv-python-headless`

你可以按需：

```bash
pip install scikit-learn
pip install torch
pip install opencv-python-headless
```

---

## 1. 目录结构（核心）

- `pr2/evidence/`：证据构造（E_t + missingness flags），支持缓存
- `pr2/posterior/`：可用性 posterior 模型、校准、阈值选择（CRC/风险控制），支持保存/加载 bundle
- `pr2/policy/`：posterior -> policy（R(w), gmax(w), lambda(w), update_mask, completion + certificates）
- `pr2/editors/`：编辑器适配层（抽象接口 + dummy 示例）
- `pr2/metrics/`：指标从 frame_log/result 中计算
- `pr2/eval/`：跑单个任务/跑任务集/聚合/可视化（含 run_meta/config snapshot）
- `configs/`：所有运行配置
- `scripts/`：常用脚本入口

---

## 2. 快速自测（跑通管线）

```bash
python -m pr2.eval.run_suite   --config configs/default.yaml   --tasks assets/demo_tasks/tier1_demo_tasks.jsonl   --out runs/demo_run
```

聚合：

```bash
python -m pr2.eval.aggregate --runs_dir runs/demo_run --out runs/demo_run/summary.json
``igg
```

---

## 3. 接入真实编辑器（最关键）

你需要实现 `pr2/editors/editor_a.py` 和/或 `pr2/editors/editor_b.py`：
- 继承 `EditorBase`，实现 `run_task(...)`
- 在你的编辑器 loop 中调用 controller 给出的 policy，并执行：
  - `update_mask`（是否允许更新）
  - `gmax`（CFG 或 guidance cap）
  - `lambda_safe`（safe prior 混合权重）
  - `radius`（trust-region 半径，用于投影：`project_l2(delta, radius)`）
  - `completion`（低可用性段的轨迹补全；不可行则 abstain）

**强制可复现要求**（论文级）：
- 每个 task 产出：
  - `result.json`（任务级指标 + 关键摘要）
  - `frame_log.jsonl`（每帧 policy + 核心中间量 + 代理指标）
- 每次 run 产出：
  - `run_meta.json`（时间戳、git commit、config hash、环境信息）
  - `config_snapshot.yaml`（冻结配置）

---

## 4. Tier-0 posterior / 阈值选择（脚手架已给）

posterior bundle 保存/加载位于：
- `pr2/posterior/bundle.py`

阈值选择（风险控制）位于：
- `pr2/posterior/crc_threshold.py`

你可以用 `scripts/02_train_posterior.py`、`scripts/03_select_tau.py` 作为模板。

---

## 5. 论文级最小实验清单（建议）

- Tier-0：posterior 校准 + 风险覆盖（ECE/Brier/Risk–Coverage）
- Tier-1（Editor-A）：Vanilla / HardGate / LossOnly / Full
- Stress：噪声 0/1/2/3 档，展示尾部灾难随噪声的斜率差异
- Q3 Transfer：Editor-B 上抽子集跑 Vanilla vs Full（同一套 policy 参数尽量不改）

