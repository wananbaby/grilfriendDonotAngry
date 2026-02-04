# grilfriendDonotAngry 添加合作者 请留言

# 基于 CV 和 NLP 的预防女朋友生气的实时检测项目

采用 Android 端深度学习模型 + 麦克风 + 摄像头，实时监测对话、语气和表情，尽早识别“吵架 / 生气”状态，并根据危险度给出提醒。

## 背景和目标

和女朋友吵架在感情中很常见，本项目希望通过多模态情绪识别，做到：

- 尽早发现“情绪升级”迹象
- 区分“有实际问题”还是“单纯想骂你发泄”
- 评估当前“生气危险度”，提醒你适当怂一点

## 算法设计

- 表情识别（CV）
  - 通过人脸图像识别情绪（愤怒、厌烦、开心等）
  - 参考项目：https://github.com/moevis/what-is-my-girlfriend-thinking
- 对话情感分析（NLP）
  - 对聊天文本做情感分析，识别负面语义和攻击性语言
  - 参考项目：https://github.com/CasterWx/python-girlfriend-mood
- 语音情感分析
  - 分析语速、音量、语调、停顿等声学特征
  - 从语气中判断情绪强度和变化趋势
- 吵架状态分类
  - 使用传统分类器（例如 SVM）或简单线性模型
  - 综合语音和文本情感得分，判断是否进入“吵架态”
- 吵架原因分析
  - 判断是真有实际问题，还是单纯想骂你发泄
  - 可引入“偏见模型”等先验，结合上下文历史
- 生气危险度识别
  - 根据语音、文本、表情多模态特征，输出危险等级

## 工程实现

- 端侧部署
  - 基于移动端深度学习模型进行训练和推理
  - 优先在本地处理，保护隐私
- 实时采集
  - 麦克风：实时获取对话语音
  - 摄像头：可选项，进行表情识别（一直举着手机拍人后果自负）
- 提示方式
  - 微信消息、通知栏、蓝牙耳机/扬声器提示
  - 尽量“温和提醒”，不要火上浇油
- 拓展服务（带一点玩笑风格）
  - 自动清空购物车服务：通过爬虫清空女朋友账号购物车，降低生气时的冲动消费风险（是否会被打更惨不在讨论范围内）
  - 雪上加霜服务：理论上可以设计让她更生气的策略，但这属于“作死高级玩法”，本项目原则上不推荐

## 当前进度（本地可训练功能）

### 1. 音频情感分类（SVM）

代码位置：

- 特征提取与数据加载：`training/audio_features.py`
- 训练脚本：`scripts/train_audio_svm.py`

数据集约定：

- 目录结构：`data_root/类别名/*.wav`
  - 示例：`data/gf_audio/angry/*.wav`、`data/gf_audio/neutral/*.wav`
- 支持格式：wav / flac / ogg / mp3（建议统一采样率 16kHz）

特征：

- MFCC（均值 + 标准差）
- 过零率、谱带宽、谱对比度、谱滚降（rolloff）
- 粗略节拍 / 语速特征

训练示例：

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

venv\Scripts\python.exe scripts\generate_dummy_audio_dataset.py --output_dir data\demo_audio --samples_per_class 30
venv\Scripts\python.exe scripts\train_audio_svm.py --data_dir data\demo_audio --output_path models\audio_svm.pkl
```

输出：

- 终端打印分类报告（precision / recall / f1）
- 模型文件：`models/audio_svm.pkl`
- 标签映射：`models/audio_svm_labels.json`

### 2. 图片表情识别（CNN）

代码位置：

- 图片数据集与 DataLoader：`training/image_dataset.py`
- 训练脚本：`scripts/train_image_cnn.py`

默认使用的数据目录结构为：

- `data/valid/0/*.jpg`
- `data/valid/1/*.jpg`
- `data/valid/2/*.jpg`
- ...

每个子目录被当作一个类别。你也可以换成自己的数据集，如：

- `data/gf_faces/angry/*.jpg`
- `data/gf_faces/happy/*.jpg`
- `data/gf_faces/neutral/*.jpg`

训练示例（PowerShell）：

```powershell
.\venv\Scripts\activate
.\venv\Scripts\python.exe scripts\train_image_cnn.py `
  --data_root data\valid `
  --epochs 1 `
  --batch_size 64 `
  --image_size 112 `
  --output_path models\image_cnn_valid.pth `
  --device cpu `
  --model simple
```

脚本会自动：

- 使用 `torchvision.datasets.ImageFolder` 加载图片数据集
- 划分训练集 / 验证集（默认 8:2）
- 使用一个简单的 3 层卷积网络（Conv + BN + ReLU + MaxPool）
- 在终端打印每个 epoch 的 loss 和 acc
- 输出模型权重和标签映射：
  - `models/image_cnn_valid.pth`
  - `models/image_cnn_valid_labels.json`

示例训练日志（CPU，SimpleCNN，epochs=1）：

```text
Epoch 1: train_loss=3.4387, train_acc=0.3018, val_loss=1.5676, val_acc=0.4046
Saved image model to models\image_cnn_simple_cpu_test.pth
```

示例推理输出（使用 `scripts/test_image_infer.py`，batch_size=4，device=cpu）：

```text
Batch size: 4
Num classes: 7
Sample 0: pred=3 (idx=3, prob=0.3419), true=4 (idx=4)
Sample 1: pred=6 (idx=6, prob=0.2013), true=4 (idx=4)
Sample 2: pred=3 (idx=3, prob=0.3709), true=3 (idx=3)
Sample 3: pred=3 (idx=3, prob=0.4278), true=4 (idx=4)
```

### 3. GPU 支持

训练脚本会自动选择设备：

- 如果 `torch.cuda.is_available()` 为真，则使用 GPU（cuda）
- 否则退回 CPU

要使用 GPU，需要：

- 机器有 NVIDIA 显卡
- 安装正确的显卡驱动和 CUDA 运行时
- 安装对应 CUDA 版本的 GPU 版 PyTorch

当前脚本默认逻辑：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

环境配置好以后，无需修改命令，训练会自动跑在 GPU 上。

## 难点与挑战

- 数据采集
  - 真实吵架场景的数据极少且非常敏感
  - 数据采集需要充分的隐私保护和双方知情同意
- 模型设计
  - 如何区分“我真有错”与“我只是想骂你”
  - 多模态信息（语音、文本、表情）的对齐和融合
- 工程落地
  - 行为需要足够隐蔽，不要反向点燃对方情绪
  - 推理延迟、功耗控制、隐私合规都需要考虑

欢迎一起完善数据集和模型结构，把这个“防止女朋友生气”的脑洞项目真正跑起来。
