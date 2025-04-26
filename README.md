# 模型蒸馏实验：DeBERTa-v3-base 到 TinyBERT

本项目旨在比较不同的知识蒸馏方法，将大型教师模型 (`microsoft/deberta-v3-base`) 的知识迁移到小型学生模型 (`huawei-noah/TinyBERT_General_6L_768D`)。

## 模型信息

*   **教师模型 (Teacher Model):**
    *   ID: `microsoft/deberta-v3-base`
*   **学生模型 (Student Model):**
    *   基础模型 ID: `huawei-noah/TinyBERT_General_6L_768D`
    *   实验中使用了两种初始化策略：
        1.  **使用6层教师模型初始化:** 从教师模型的特定层初始化学生模型。
        2.  **TinyBert初始化:** 使用预训练的 `TinyBERT_General_6L_768D` 权重进行初始化。

## 层匹配策略

当使用教师模型初始化学生模型时，采用以下层匹配策略：

*   **教师模型层 (Teacher Layers):** `[0, 2, 4, 6, 8, 10]`
*   **学生模型层 (Student Layers):** `[0, 1, 2, 3, 4, 5]`

即，教师模型的第 0 层权重用于初始化学生模型的第 0 层，教师模型的第 2 层权重用于初始化学生模型的第 1 层，以此类推。

## 实验设置与结果

我们在 `imdb` 和 `sst2` 数据集上进行了实验，对比了传统蒸馏方法和 Patient-Knowledge-Distillation 方法。详细的超参数设置和准确率结果如下表所示：

| 方法 (Method)                 | 教师模型 (Teacher) | 学生模型初始化 (Student Init) | 超参数 (Hyperparameters)                                                                                                                                                              | 数据集 (Dataset) | 准确率 (Accuracy) |
| :---------------------------- | :----------------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------- | :---------------- |
| 传统蒸馏方法                  | deberta-v3-base    | 使用6层教师模型初始化         | epochs=3, batch_size=16, lr=3e-5, warmup_ratio=0.1, weight_decay=0.01, **蒸馏特定参数:** alpha=0.5, temperature=4.0                                                                     | imdb             | 0.8363            |
| 传统蒸馏方法                  | deberta-v3-base    | TinyBert初始化                | epochs=3, batch_size=16, lr=3e-5, warmup_ratio=0.1, weight_decay=0.01, **蒸馏特定参数:** alpha=0.5, temperature=4.0                                                                     | imdb             | 0.9190            |
| 传统蒸馏方法                  | deberta-v3-base    | 使用6层教师模型初始化         | epochs=3, batch_size=32, lr=5e-5, warmup_ratio=0.1, weight_decay=0.01, **蒸馏特定参数:** alpha=0.5, temperature=4.0                                                                     | sst2             | 0.89449           |
| 传统蒸馏方法                  | deberta-v3-base    | TinyBert初始化                | epochs=3, batch_size=32, lr=5e-5, warmup_ratio=0.1, weight_decay=0.01, **蒸馏特定参数:** alpha=0.5, temperature=4.0                                                                     | sst2             | 0.9128            |
| Patient-Knowledge-Distillation | deberta-v3-base    | 使用6层教师模型初始化         | epochs=3, batch_size=32, lr=5e-5, warmup_ratio=0.1, weight_decay=0.01, **蒸馏特定参数:** alpha=0.5, temperature=4.0, **beta=1**                                                          | imdb             | 0.9270            |
| Patient-Knowledge-Distillation | deberta-v3-base    | 使用6层教师模型初始化         | epochs=3, batch_size=32, lr=5e-5, warmup_ratio=0.1, weight_decay=0.01, **蒸馏特定参数:** alpha=0.5, temperature=4.0, **beta=1**                                                          | sst2             | 0.9037            |

**注:**
*   `alpha`: 控制蒸馏损失（KL散度）和学生模型自身任务损失（如交叉熵）之间的权重。
*   `temperature`: 在计算 softmax 时用于平滑教师和学生模型的输出概率分布。
*   `beta`: Patient-Knowledge-Distillation 方法中可能引入的额外损失项权重（例如中间层表示的匹配损失）。

## 结论（初步）

*   **初始化策略:** 使用预训练的 TinyBert 权重进行初始化（`TinyBert初始化`）通常比直接从教师模型的几层进行初始化（`使用6层教师模型初始化`）在传统蒸馏方法下效果更好。
*   **蒸馏方法:** Patient-Knowledge-Distillation 方法在 `imdb` 数据集上，使用教师层初始化时，取得了比传统蒸馏方法更高的准确率。
*   **数据集:** 模型在 `imdb` 数据集上的准确率普遍高于 `sst2` 数据集。