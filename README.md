# 实验一：十类英文文本分类

本仓库包含实验数据、模型选择代码、验证结果、预测文件和[实验报告](实验一报告.md)。任务是根据英文新闻或讨论组文本预测 0～9 的类别。测试数据没有标签，因此本文中的准确率和 F1 都是从有标签训练数据中划出的**验证集成绩**，不是最终测试成绩。

## 文件说明

| 路径 | 内容 |
| --- | --- |
| `data/train_data.csv` | 7,368 条带标签文本，列为 `text,target` |
| `data/test_data_unlabeled.csv` | 2,457 条无标签文本，列为 `text` |
| `run_experiment.py` | 固定划分，比较 88 组文本处理、TF-IDF 和分类器配置 |
| `run_stemming_comparison.py` | 在已选线性 SVM 配置上比较有无词干，并训练最终模型 |
| `run_mlp_comparison.py` | 使用相同词干特征比较三组 MLP |
| `check_reliability.py` | 五次分层划分与重复文本检查 |
| `third_party/snowball_english/` | 最终模型使用的 Snowball 英文词干实现及许可证 |
| `experiment_results/baseline/` | 首轮 88 组验证结果与未使用词干的预测 `predictions.csv` |
| `experiment_results/stemming/` | 词干对照、合并后的 90 组结果及最佳配置 |
| `experiment_results/mlp/` | MLP 验证结果和模型选择摘要 |
| `experiment_results/reliability/` | 多次划分的成绩、配对差异与重复文本检查结果 |
| `prediction.csv` | **最终选定模型的预测**，位于仓库根目录，单列、无表头，共 2,457 行 |
| `实验一报告.md` | 实验设计、模型比较、误差分析与结论 |
| `report_assets/` | 报告引用的两张最终 PNG 图片 |

## 环境与复现

使用 Python 3.10。实验时使用的主要依赖版本列于 `requirements.txt`；可在仓库根目录安装：

```bash
python -m pip install -r requirements.txt
```

然后按以下顺序运行：

```bash
python run_experiment.py --mode full
python run_stemming_comparison.py
python run_mlp_comparison.py
python check_reliability.py
```

第一步将带标签数据按类别分层划分为 5,894 条训练和 1,474 条验证，随机种子固定为 42；比较朴素贝叶斯、逻辑回归和线性 SVM 的 88 组配置，并将结果与首轮预测写入 `experiment_results/baseline/`。第二步读取完整的 88 组结果，复核无词干对照，再比较两组新增的词干配置；依据验证集 macro-F1 选型，用全部 7,368 条带标签数据重训，将对照结果写入 `experiment_results/stemming/`，并生成根目录的最终提交文件 `prediction.csv`。因此无需单独的最终模型脚本。

第三步是额外的 MLP 对照：三组 MLP 均未超过已选线性 SVM，最终预测仍使用 `prediction.csv`。第四步检查固定模型在五次划分上的表现和完全重复文本，不改变提交预测。`run_experiment.py --mode quick` 只是快速检查，会覆盖 `experiment_results/baseline/`；若要继续运行词干对照，须先执行上述 `--mode full`。

## 最终方案与报告

最终方案清理指定邮件头，过滤英文停用词，使用 Snowball 英文词干和单词/双词 TF-IDF（`max_features=50000`、`min_df=2`、`sublinear_tf=True`），分类器为 `LinearSVC(C=0.5)`。固定种子 42 的验证准确率为 **0.9484**，验证 macro-F1 为 **0.9488**。无标签测试集的真实准确率未知。根目录只保留最终提交用的 `prediction.csv`；首轮预测保存在 `experiment_results/baseline/predictions.csv`。

[实验一报告.md](实验一报告.md) 使用下面两张图片；图片已按报告中的相对路径放置，可直接在支持 Markdown 的页面查看：

- [参数搜索、词干增益与学习曲线](report_assets/figure_1_selection_reliability.png)
- [分类误差与特征分析](report_assets/figure_2_errors_features.png)

本仓库保存最终图片，不包含图片生成脚本或报告排版过程文件。
