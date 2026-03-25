# Regional Traffic Signal Optimization with Transformer

一个从 0 到 1 的区域交通信号优化项目：

- 使用公开交通流量数据（UCI Metro Interstate Traffic Volume）
- 构建 2x2 路网（4 路口）离散时间仿真器
- 使用时空 Transformer 学习信号控制策略
- 与固定配时、Max-Pressure 基线进行本地仿真对比

## 1. 项目结构

```text
tranformmer/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ outputs/
├─ scripts/
│  └─ run_pipeline.py
├─ src/traffic_transformer/
│  ├─ config.py
│  ├─ dataset.py
│  ├─ simulator.py
│  ├─ model.py
│  ├─ training.py
│  └─ evaluate.py
└─ requirements.txt
```

## 2. 环境与安装（Windows PowerShell）

```powershell
D:/Anaconda/Scripts/conda.exe create -y -n trans python=3.12
D:/Anaconda/Scripts/conda.exe run -n trans python -m pip install -r d:\Desktop\tranformmer\requirements.txt
```

如果你当前的 `trans` 路径是 `D:\Anaconda\pkgs\trans`，也可直接：

```powershell
& "D:\Anaconda\pkgs\trans\python.exe" -m pip install -r "d:\Desktop\tranformmer\requirements.txt"
```

## 3. 一键运行（下载+训练+评估）

```powershell
& "D:\Anaconda\pkgs\trans\python.exe" "d:\Desktop\tranformmer\scripts\run_pipeline.py"
```

运行后会在 `outputs/` 看到：

- `transformer_policy.pt`：训练好的模型
- `evaluation_results.json`：三种策略指标对比
- `evaluation_plot.png`：结果图

## 4. 核心方法说明

### 4.1 仿真环境

- 每个路口有两条冲突方向（NS / EW）队列
- 每个时间步只能放行一个方向，放行能力为 `service_rate`
- 部分车流按 `transfer_ratio` 进入下游路口，实现区域耦合

### 4.2 基线策略

- Fixed-Time：固定周期交替放行
- Max-Pressure：选择当前队列压力更大方向放行

### 4.3 Transformer 策略

- 输入：最近 `history_steps` 个时刻、所有路口队列状态
- 编码：时间嵌入 + 路口嵌入 + Transformer Encoder
- 输出：每个路口四类离散动作，完美解决1秒级决策抖动并实现**动态延长时间**：
  - `0`: 南北向绿灯，持续 10秒 (短绿灯)
  - `1`: 南北向绿灯，持续 20秒 (长绿灯)
  - `2`: 东西向绿灯，持续 10秒 (短绿灯)
  - `3`: 东西向绿灯，持续 20秒 (长绿灯)
- 优点：模型不仅接管了对相位的决策，同时还接管了绿灯持续时间的自适应控制。配合 SUMO 仿真器可以严格执行所下发的周期，从根源上杜绝了每秒都在做改变产生的过频振荡 (Flickering)。

### 4.4 训练方式

- 专家数据生成：升级版 Max-Pressure 不仅输出相位，更基于双向排队压力差（`diff > 5` 则为长绿灯）输出时长标签。
- 通过监督学习（行为克隆）训练 Transformer。
- 损失函数：四分类交叉熵。

## 5. 已验证结果（当前一次运行）

来自 `outputs/evaluation_results.json`：

- Fixed-Time
  - `avg_queue`: 5322.64
  - `throughput`: 6958.45
- Max-Pressure
  - `avg_queue`: 5016.67
  - `throughput`: 7570.48
- Transformer
  - `avg_queue`: 5016.79
  - `throughput`: 7570.30

结论：Transformer 基本复现了 Max-Pressure 的控制效果，显著优于固定配时。

## 6. 用到的框架

- **PyTorch**：Transformer 模型定义与训练
- **NumPy / Pandas**：数据处理、时序重采样、数组计算
- **Matplotlib**：结果可视化
- **scikit-learn**：项目依赖保留（便于后续扩展评估）
- **Requests**：在线下载公开数据集

## 7. 后续可扩展方向

- 引入强化学习（PPO / SAC）做在线策略优化
- 替换为真实路网与信号相位约束（黄灯、全红）
- 加入多目标优化（延误、排放、公平性）
- 对接 SUMO / CityFlow 做更高保真仿真
