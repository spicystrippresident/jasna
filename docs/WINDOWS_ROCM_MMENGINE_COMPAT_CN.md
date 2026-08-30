# Windows ROCm MMEngine 导入兼容层

## 目的与边界

Windows ROCm 的 Torch wheel 不提供 distributed c10d，
`torch.distributed.is_available()` 因此为 false。MMEngine 0.10.x 即使只用于
BasicVSR++ 单进程推理，也会在导入阶段访问 `ReduceOp` 和 FSDP 模块。

Jasna 只在 `win32 + torch.version.hip + distributed unavailable` 的精确交集内补齐
MMEngine 导入所需的两个表面：`ReduceOp` 占位符和不可实例化的 FSDP 模块。
FSDP 构造仍会显式报错，且 `torch.distributed.is_available()` 保持 false。

Linux ROCm、Windows/Linux CUDA，以及真正具备 distributed 的 Torch 环境完全不进入
该兼容层。本修复只解除 MMEngine 的模块导入阻断，不启用分布式训练，不改变
BasicVSR++ 模型、checkpoint、推理路线、共享编排或产品默认值。

## 实现位置

- `jasna/models/basicvsrpp/mmengine_compat.py`：精确平台能力判定与临时导入表面。
- `jasna/models/basicvsrpp/__init__.py`：仅在加载 BasicVSR++ 模型前应用兼容层。

## 验证

聚焦测试覆盖：

- Windows ROCm 且 distributed 不可用时可完成 MMEngine 导入；
- FSDP 构造仍失败；
- Linux ROCm、CUDA 和具备 distributed 的环境不注入兼容层；
- 临时模块状态在测试后恢复。

运行命令：

```bash
python -m pytest -q tests/test_mmengine_windows_rocm_compat.py
```

这项 Linux mock/结构验证不能替代 Windows ROCm 真机加载 BasicVSR++ 的验收。
