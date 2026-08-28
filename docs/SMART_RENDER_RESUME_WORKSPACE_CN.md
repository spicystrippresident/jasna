# Smart Render 可恢复工作区与完整处理签名

更新时间：2026-08-29

Smart Render 的片段工作区改为固定、可恢复且由完整处理签名绑定。取消、进程异常或
机器重启后，下一次只复用同一输入、同一输出目标、同一 splice plan、同一模型和同一
处理设置产生的完整片段。

## 工作区合同

- 路径由输出名和 canonical signature SHA-256 派生，不依赖一次性临时目录。
- manifest 采用原子 JSON 写入；`running` 状态在下一次打开时回退为 `pending`。
- complete artifact 记录绝对路径、size、mtime 和完整 SHA-256；任一不符即拒绝复用。
- source identity 包含绝对路径、size、mtime 以及首尾各 1 MiB 的 hash。
- model identity 包含 detection、BasicVSR++ checkpoint 和 LUT 的 size、mtime、完整 hash。
- splice identity 包含 time base、start/end PTS、keyframes、B-frame 合同、decode delay 和
  每个 copy/render span。
- processing identity 由 session factory 构造，包含 Jasna 版本、设备、FP16、B4/B8、
  detector 阈值、Tracker/clip 参数、去噪、Primary TensorRT、完整 secondary 设置、VR、
  sharpen 与高帧率 retarget。
- encoding identity 包含 codec 和已解析的 encoder settings；算法版本为
  `jasna-smart-render-workspace-v3`。

成功路径先验证临时成片的 codec、duration 和尾部可读性，再 fsync、同目录原子替换、
复验最终文件，最后才删除工作区。取消和一般失败保留证据及可复用片段；HEVC seam
合同本身被判定不兼容时清理该签名工作区，防止反复复用已知无效组合。

## 验收

- manifest 损坏会保留 invalid backup 并重建；
- source/settings/model/algorithm 任一变化会得到不同工作区；
- complete hash 被篡改后不会复用；
- path traversal 和 workspace 外 artifact 会被拒绝；
- 已完成 render span 只增加进度，不制造虚假速度样本；
- TVAI 与 RTX secondary 的全部有效参数都进入签名；
- Linux 聚焦回归：194 passed、1 skipped。
