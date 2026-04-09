# PETS - Probabilistic Ensemble Trajectory Sampling

## 项目概述
PETS是一种基于模型的强化学习算法，使用概率集成模型来预测环境动态，并通过轨迹采样来优化策略。本项目实现了PETS算法的核心功能，适用于连续控制任务。

## 环境要求
- Python 3.9+
- numpy>=1.21.0
- scipy>=1.7.0
- gym>=0.21.0
- torch>=1.10.0
- matplotlib>=3.4.0

## 安装
```bash
git clone <仓库URL>
cd 05_PETS
pip install -r requirements.txt
```

## 使用方法
```bash
# 运行PETS算法
python pets_main.py
```

## 项目结构
```
├── pets_main.py       # 主运行文件
├── cem.py            # CEM优化器实现
├── models.py         # 概率集成模型实现
├── replay_buffer.py  # 经验回放缓冲区
├── fake_env.py       # 模拟环境
├── requirements.txt  # 项目依赖
├── README.md         # 项目说明
└── LICENSE           # 许可证
```

## 核心功能
- **概率集成模型**：使用多个神经网络模型集成，提供不确定性估计
- **CEM优化器**：通过交叉熵方法优化轨迹
- **经验回放**：存储和采样历史经验
- **模拟环境**：用于测试算法性能

## 贡献指南
欢迎提交Issue和Pull Request来改进项目。

## 许可证
本项目使用 MIT 许可证。