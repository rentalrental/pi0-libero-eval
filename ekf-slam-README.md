# 无人方程式赛车：锥桶感知 + EKF SLAM 实时定位与建图

> Formula Student Driverless 赛车的**完整感知 → 定位建图链路**：
> ZED 双目相机 + **YOLO 实例分割**检测彩色锥桶并测距，再用 **EKF SLAM**（轮速里程计 + 转向角驱动预测，
> 锥桶观测更新）在线维护车辆位姿 + 全局锥桶地图。
> SLAM 实现严格对照 **Thrun《Probabilistic Robotics》(2005) Table 10.2（EKF SLAM, unknown correspondence）**。

> 👤 **项目与署名说明：** 团队项目（UWE Formula Student AI，`ads-ws` 工作区）。
> 本仓库聚焦**我个人负责的部分：锥桶感知（YOLO 分割 + ZED 双目测距）+ EKF SLAM 节点 + 数据关联**。
> CAN 通信、控制、车辆仿真等由车队其他模块提供。**本方案纯视觉，未使用 LiDAR。**

---

## 1. 这个项目展示了什么

| 能力 | 我做了什么 |
|---|---|
| **锥桶感知** | ZED 双目 RGB + **YOLO 实例分割（v11m）**：输出锥桶类别（蓝/黄/橙/大橙）+ 掩码 |
| **双目测距** | 取掩码内 ZED 深度点 → 滤 NaN → 取 **3D 中位数（median）** 抗离群 → 锥桶坐标（纯视觉，无 LiDAR） |
| **相机标定** | 内参（张正友棋盘格法）+ 外参（R, t），把锥桶从相机系转到车体/世界系 |
| **EKF SLAM** | unknown-correspondence EKF SLAM（Thrun 表 10.2）：预测步 + 观测更新，在线维护位姿与协方差 |
| **状态表示** | `mu = [x, y, θ, lm₁ₓ, lm₁ᵧ, …]`，预分配 **1500 个 landmark**，位姿块 + landmark 块联合维护 |
| **数据关联** | **颜色门控（硬预筛）+ 马氏距离 + 卡方门限**：先按颜色过滤候选，再马氏距离匹配，防蓝/黄混淆 |
| **ROS2 工程** | rclpy 节点，订阅轮速 + 锥桶检测，100 Hz 发布位姿 / 全局地图 / 历史轨迹 |

## 2. 方法

### 2.0 感知前端（纯视觉，无 LiDAR）
- **检测**：ZED 双目 RGB 图 → YOLO 实例分割（v11m）→ 锥桶类别 + 像素掩码。
- **测距**：在掩码区域内取 ZED 点云深度点，**滤除 NaN**，取 **3D 中位数**作为锥桶位置——中位数对边缘/离群深度点比均值更鲁棒。
- **标定**：相机内参用棋盘格（张正友法）标定，外参 `(R, t)` 把锥桶坐标转入车体系。
- **输出**：每个锥桶的 `(range, bearing, colour)`，仅保留 **≤ 10 m** 的近距观测喂给 SLAM。

### 2.1 预测步（运动模型）
- 输入 `u = [v, ω]`：后轮 RPM → `v = 2π·(轮径0.505/2)·(rpm/60)`（速度上限 15 m/s）；
  转向角 + 轴距 1.53 → 自行车模型 `r = 轴距/tan(转角)`，`ω = v/r`。
- 速度运动模型推进均值 `mu_bar = g(mu, u)`，构造位姿雅可比 `G`，嵌入全状态雅可比 `F_x`。
- 协方差传播：`sigma_bar = F_x·sigma·F_xᵀ + F_xT·R·F_xT'`，过程噪声 **R 只加在 3×3 位姿块**上。
  - 运动噪声 `R = diag([1.5, 1.5, 1.5°])²`；车静止时切换到 `R_STATIC = diag([0.01, 0.01, 0.01°])²`，抑制零速漂移。

### 2.2 数据关联（核心，防错配）
对每个锥桶观测：
1. **颜色门控**：只把**同色**（或任一方为 unknown=4）的 landmark 作为候选——异色直接跳过，防几何相近时蓝黄锥桶混淆。
2. **马氏距离**：`S = H·sigma_bar·Hᵀ + Q`，`dist = innov·S⁻¹·innov`；取最小且 `< ALPHA` 的为匹配。
   - 观测噪声 `Q = diag([1.0, 0.75°])²`；门限 `ALPHA = 3`（2 自由度卡方 95% 置信门）。
3. **无匹配 → 新建 landmark**，初始协方差 `1000·I`；已匹配但原色 unknown 的，升级为确定颜色。

### 2.3 更新步
标准 EKF 更新：`innov = z − ẑ`；`K = sigma_bar·Hᵀ·S⁻¹`；`mu += K·innov`；`sigma = (I − K·H)·sigma_bar`。
测量雅可比 `H` 按 Thrun Eq. (10.14) 对 range-bearing 模型求导。

> 锥桶颜色约定：Blue=0 / Yellow=1 / Orange=2 / Big-Orange=3 / Unknown=4。
> 起终点判断靠**大橙色锥桶**。属于 unknown correspondence（开放地图匹配），未做显式回环（loop closure）。

## 3. 结果 / Demo

- **感知**：锥桶检测距离约 **30 m**（较基线提升约 30%）；中位数测距对离群深度点鲁棒。
- **定位建图**：定位误差收敛至**厘米级**，随圈数累计出稳定的全局锥桶地图。
- 在线话题输出：`/uwe_ai/slam/car_pose`、`/uwe_ai/slam/map`、`/uwe_ai/slam/pose_history`。
- 📹 **[填: 放一段 rviz 实时建图录屏 GIF/视频]**（素材在你另一块硬盘上）—— 展示车辆轨迹 + 彩色锥桶地图随赛道逐步建立，是这个项目最抓眼的一屏。

## 4. 技术栈

`Python` · `ROS2 Humble` · `NumPy` · `YOLO（v11m 实例分割）` · `ZED 双目` · `OpenCV（相机标定）` · `EKF SLAM` · `rviz2`

## 5. 如何运行

ROS2 Humble 工作区（团队仓库 `UWE-FSAI/ads-ws`）：

```bash
# 构建
colcon build --packages-select slam_node bristol_msgs ads_dv_msgs
source install/setup.bash

# 启动 SLAM（或用 launch_files/slam_launch.py 一起拉起感知 + SLAM）
ros2 run slam_node slam_node
```

**接口**
- 订阅：`/VCU2AI/wheel_speeds`（`WheelSpeeds`）、`/perception/cone_detections`（`ConeArrayWithCovariance`）
- 发布：`/uwe_ai/slam/car_pose`、`/uwe_ai/slam/map`、`/uwe_ai/slam/pose_history`

## 6. 代码亮点（我可以深挖讲的点）

- **纯视觉锥桶测距用 3D 中位数而非均值**：掩码边缘的深度点容易是离群值，中位数抗离群、更稳。
- **颜色门控 + 马氏距离 + 卡方门**三级数据关联，比单纯欧氏距离更鲁棒，专门解决蓝/黄锥桶几何相近的错配。
- **为什么用滤波派（EKF）而非因子图/BA**：FS 赛道锥桶有限、需在线实时，EKF 单帧更新足够且延迟低。
- **R/R_STATIC 双噪声**：静止切小噪声抑制零速漂移，是实跑里调出来的工程细节。
- **预分配 1500 landmark** 的定长状态向量 + 协方差，避免在线频繁扩容。

> 实现对照：EKF SLAM = Thrun 表 10.2；测量雅可比 = Eq. (10.14)。源码见 `src/SLAM/slam_node/slam_node/slam_node.py`。
