# L3GSA experiments

用于维护L3GSA论文实验的环境配置、仿真代码、数据来源和可复现结果。
平台配置以当前英文稿为准；平台实测、吞吐标定仿真和建模假设分别记录。

| 内容 | 入口 |
|---|---|
| 平台与参数来源 | [实验环境](docs/experiment-environment.md) |
| 子图(a)：混合需求下的请求分配 | [实验说明](experiments/task_distribution/README.md) |
| 当前结果及解释范围 | [结果说明](experiments/task_distribution/RESULTS_AND_INTERPRETATION.md) |
| 结果图 | [PNG](experiments/task_distribution/output/allocation_ratio.png) / [PDF](experiments/task_distribution/output/allocation_ratio.pdf) |

## 快速复现

需要Python 3.12。首次安装依赖后，运行以下命令：

```bash
cd experiments/task_distribution
python -m pip install -r requirements.txt
python simulate_allocation.py --sensitivity
python validate_model.py
python plot_allocation.py
```

只改绘图时，修改 `plot_allocation.py` 顶部的 `STYLE` 并重新执行绘图命令。
绘图脚本只依赖Matplotlib和CSV，不运行或导入仿真脚本，也不需要额外的样式JSON。

## 当前范围

- 子图(a)为设备prefill实测速率标定的排队仿真，不是新增硬件测试。
- 每请求400输入tokens、1输出token；模型没有复现vLLM内部batch、KV cache或真实端到端时延。
- 主图包含三组独立运行，折线为运行比例的均值，误差棒为min–max。
- 代码、汇总数据、配置和图纳入Git；大体积逐请求轨迹由脚本生成。
- 后续实验通过独立分支和Pull Request提交；本仓库不自动部署应用或操作GPU环境。
