from __future__ import annotations

"""
在这里接入你的 Editor-A（per-frame diffusion editor）。

你需要实现：
- 加载视频/初始化编辑器
- 每个 frame/step 计算 proposed update (delta)
- 从 controller 拿到 policy：
  - update_mask
  - radius (trust-region)
  - gmax (guidance cap)
  - lambda_safe (safe prior mix)
  - completion（如果该帧处于 completion segment 且 feasible）
- 应用：
  - delta <- project_l2(delta, radius)
  - 若 update_mask=False，则不更新（或只更新 safe prior 部分，取决于你定义）
  - 记录 frame_log

TODO: Implement your real adapter here.
"""
