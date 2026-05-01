# 本地视觉管线运行摘要

- 输入视频：`/home/cells/ai/data/input/demo.mp4`
- 输出视频：`/home/cells/ai/outputs/demo_side_by_side.mp4`
- 性能 CSV：`/home/cells/ai/outputs/performance.csv`
- 已处理帧数：120
- 决策结果：

风险等级：低。
最近目标：person，伪距离 Z≈11.28，未位于画面中心通道。
建议：障碍物距离相对较远，维持当前速度并持续监测即可。
说明：当前结果基于相对深度 Pseudo-LiDAR 和本地规则，尚未调用云端 VLM。

说明：当前深度是相对深度映射出的 Pseudo-LiDAR 伪距离，并非真实米制深度。
