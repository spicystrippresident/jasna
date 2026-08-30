# Linux AMD AMF dma-buf cache FD 生命周期

更新时间：2026-08-30

## 功能边界

本 PR 只收紧已经启用的 Linux AMD AMF stable dma-buf identity cache 的文件描述符
所有权与验收日志，不改变任何解码选择顺序或产品默认：

- Linux AMD AV1 Main NV12/P010 的 `auto` 仍按上一 PR 使用 cache；
- H.264/HEVC cache 仍不作为产品默认；
- native gate 外输入仍使用普通 PyAV；不存在第二套 Linux AMD 解码 fallback；
- Windows、NVIDIA、共享 Pipeline、RF-DETR、BasicVSR++、B4、CQ 和编码器均不变。

本 PR 不包含已经裁决为 `CLOSED_UNSUPPORTED` 的 HIP→AMF/Vulkan encode surface
ring、source digest、observer/reuse/ownership 探针或 GPU-only encoder 接入。

## FD 所有权合同

Vulkan 每次导出一个 opaque dma-buf FD 后，bridge 按 stable `(st_dev, st_ino)`
身份处理：

1. cache miss：FD 的所有权随 `hipImportExternalMemory` 成功转交给 HIP；Jasna 不再
   `close()` 该整数，session close 时销毁对应 HIP import/mapping。
2. cache hit：本次新导出的 FD 不再需要，立即且只调用一次 `close()`。
3. import 前或其他失败路径仍持有 FD 时，cleanup 只关闭一次。
4. `close()` 失败记录返回值和 `errno`，reader 在 session close 审计时 fail closed；
   Linux 上不盲目重试，避免已释放后复用的整数被误关。

reader 的最终 JSON transport stats 现在包含：

- `cache_fd_close_calls` / `cache_fd_close_failures`；
- `cache_last_fd_close_errno`；
- `cache_fd_ownership_transfers`。

成功门要求 cache hit 数等于 close 次数，cache miss 数等于 ownership transfer
次数，且 close failure 与最后 errno 都为 0。最终 stats 会以
`AMF interop transport stats reader=` 前缀写入 run log，供媒体事务独立解析。

## 已有真实 Linux 验证

同源 H.264 High/NV12 4096×2048、120 帧 canary 中两个 reader 都得到
`36 miss + 84 hit`，84 次 cache-hit close、0 次 close failure/errno，36 次 FD
ownership transfer；36/36 HIP import/destroy、fixed-context create/close 和 writer
资源全部配平。随后三轮独立 lifecycle 均通过软件 strict decode、120/120 帧、PTS、
framemd5、资源、残留进程、partial 和 kernel journal 门。该 H.264 验证只证明 cache
外壳生命周期，不把 H.264 cache 迁入产品默认。

证据目录：

```text
/media/latiao/D/AI/amf-unified-work/transactions/jasna-ubuntu-amf-h264-cache-ring-canary-b1-attempt003-20260829
/media/latiao/D/AI/amf-unified-work/transactions/jasna-ubuntu-amf-h264-cache-ring-lifecycle3-b1-aggregate-20260829
```

聚焦单元测试覆盖 cache hit/miss 计数、FD close error fail-closed 和最终 JSON stats
序列化。Linux 累计验收结果统一记录在 `docs/STACKED_PR_LINUX_ACCEPTANCE_CN.md`。
