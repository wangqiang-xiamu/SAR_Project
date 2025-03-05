# FixMatch

**FixMatch** 是一个非官方的 **PyTorch** 实现，基于论文 [*Simplifying Semi-Supervised Learning with Consistency and Confidence*](https://arxiv.org/abs/2001.07685)。

官方 **TensorFlow** 实现：[GitHub](https://github.com/google-research/fixmatch)

本代码仅支持 **FixMatch（RandAugment）**。

---

## 🚀 结果

### 📌 CIFAR-10

| 标签数 | 40 | 250 | 4000 |
|--------|----|-----|------|
| 论文（RA） | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
| 本代码 | **93.60** | **95.31** | **95.77** |
| 🔗 准确率曲线 | [链接](#) | [链接](#) | [链接](#) |

> **2020 年 11 月**：修复 EMA 问题后重新测试。

### 📌 CIFAR-100

| 标签数 | 400 | 2500 | 10000 |
|--------|----|------|-------|
| 论文（RA） | 51.15 ± 1.75 | 71.71 ± 0.11 | 77.40 ± 0.12 |
| 本代码 | **57.50** | **72.93** | **78.12** |
| 🔗 准确率曲线 | [链接](#) | [链接](#) | [链接](#) |

> **训练时使用的额外参数**：`--amp --opt_level O2 --wdecay 0.001`

---

## 📖 使用方法

### 🏋️ 训练

#### ✅ 在 CIFAR-10（4000 条标注数据）上训练模型
```
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5
```

test
```
python train.py --dataset cifar10 --num-labeled 10 --arch wideresnet --batch-size 2 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5
```

#### ✅ 在 CIFAR-100（10000 条标注数据）上训练模型（使用 DistributedDataParallel）

```
python -m torch.distributed.launch --nproc_per_node 4 ./train.py --dataset cifar100 --num-labeled 10000 --arch wideresnet --batch-size 16 --lr 0.03 --wdecay 0.001 --expand-labels --seed 5 --out results/cifar100@10000
```
参数说明
python -m torch.distributed.launch
该命令用于启动 PyTorch 的分布式训练。在此命令中，torch.distributed.launch 模块会启动多进程进行并行计算。适用于在多 GPU 环境下进行训练。此参数可以使训练过程更加高效，特别是在使用多个 GPU 时。

--nproc_per_node 4
这个参数设置了每个节点（通常是每台机器）使用的 GPU 数量。在此示例中，设置为 4，意味着在每台机器上使用 4 个 GPU 进行训练。如果你有更多的 GPU，可以增加此参数的值来充分利用资源进行并行训练。

./train.py
指定训练脚本的路径。在此示例中，训练脚本 train.py 位于当前目录下。该脚本负责加载数据集、构建模型、配置优化器，并进行模型训练。

--dataset cifar100
该参数指定了要使用的数据集类型。在此示例中，使用的是 CIFAR-100 数据集。CIFAR-100 是一个包含 100 类物体图像的小型图像数据集，通常用于图像分类任务。该数据集包含 60,000 张 32x32 像素的彩色图像。

--num-labeled 10000
指定训练时使用的标注样本数量。在此示例中，使用了 10,000 个标注样本进行训练。在半监督学习任务中，我们通常只标注数据集的一部分，剩余数据用伪标签进行训练。

--arch wideresnet
该参数指定了使用的模型架构。在此示例中，使用了 WideResNet 网络架构。WideResNet 是一种具有较宽的卷积层的网络架构，通常比普通的 ResNet 更快且具有更好的性能。适用于大规模图像分类任务。

--batch-size 16
设置训练的批次大小，即每个训练步骤中处理的样本数。此参数控制每次迭代时的内存使用量和训练的效率。较小的批次大小能够减少显存的占用，但可能会增加训练时间。在此示例中，批次大小设置为 16。

--lr 0.03
该参数设置学习率（learning rate）。学习率控制优化器更新模型参数的步长。较小的学习率可能导致训练过程缓慢，而较大的学习率可能使模型在训练过程中震荡，导致难以收敛。在此示例中，学习率设置为 0.03。

--wdecay 0.001
权重衰减（weight decay），通常用于正则化，以防止模型过拟合。通过对模型参数施加惩罚，权重衰减能有效地防止模型的复杂度过高。在此示例中，设置为 0.001，表示对权重进行轻微的正则化。

--expand-labels
该选项启用标签扩展功能。在半监督学习中，通常会通过某种方法（如伪标签）扩展数据集中的标签，以利用未标注的数据进行训练。启用此选项后，模型会尝试生成未标注数据的伪标签，并将其用于训练。

--seed 5
设置随机种子，以确保训练的可重现性。设置相同的种子值可以保证每次训练的初始化参数和数据处理顺序相同，从而保证训练结果的一致性。在此示例中，种子值设置为 5。

--out results/cifar100@10000
该参数指定训练结果的输出目录。在此示例中，训练结果将保存到 results/cifar100@10000 目录下，包括训练日志、模型检查点以及其他相关文件。你可以根据需要修改输出路径以保存到不同位置。



### 📊 监控训练进度

使用 TensorBoard 可视化训练日志：
在 TensorBoard 中，检查以下内容：
Scalars 选项卡：查看 train/loss、train/accuracy、test/loss 和 test/accuracy 的曲线。
Graphs 选项卡：如果记录了模型结构，可以查看计算图。
Histograms 选项卡：如果记录了权重或梯度分布，可以查看直方图。

tensorboard --logdir=<your_out_dir>

```bash
tensorboard --logdir=results/cifar10@4000.5
```

---

## 📌 依赖环境
- Python 3.6+
- PyTorch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex（可选）

---

## 🔗 相关实现
- [Meta Pseudo Labels](https://github.com/kekmodel/Meta-Pseudo-Labels)
- [UDA for images](https://github.com/kekmodel/UDA-pytorch)

---

## 📚 参考资料
- **FixMatch 官方 TensorFlow 实现**：[GitHub](https://github.com/google-research/fixmatch)
- **MixMatch 非官方 PyTorch 实现**
- **RandAugment 非官方 PyTorch 复现**
- **PyTorch 图像模型**

---

## 📜 引用
```bibtex
@misc{jd2020fixmatch,
  author = {Jungdae Kim},
  title = {PyTorch implementation of FixMatch},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kekmodel/FixMatch-pytorch}}
}
