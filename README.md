# NeuralODE-Identification
## 说明
本项目基于PyTorch开发，主要实现了基于神经常微分方程的飞行器动力学模型辨识。

## 结构
- `data_processing.py`：数据预处理模块
- `flight_scaler.py`：数据归一化模块
- `NeuralODEFunc.py`：神经网络、物理模型及ODE模块
- `train.py`：训练及参数设置模块
- `test.py`：测试模块（未完成）
- `generate_data.py`：测试数据生成

## 依赖
- Python 3.9.22
- PyTorch 2.7.1