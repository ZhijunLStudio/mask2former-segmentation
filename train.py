from PIL import Image
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, TrainingArguments, Trainer
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, image_size=(384, 384)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.image_size = image_size
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # 加载图像和掩码
        image = Image.open(image_path).convert("RGB").resize(self.image_size)
        mask = Image.open(mask_path).convert("L").resize(self.image_size)
        
        # 转换掩码格式
        mask = np.array(mask)
        mask = np.where(mask == 255, -1, mask)  # 背景：-1，类别：0
        mask_labels = torch.tensor(mask, dtype=torch.int64)

        # 图像预处理
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,  # 保留在 CPU
            "mask_labels": mask_labels,    # 保留在 CPU
        }

# 加载数据集
def load_datasets(dataset_root, processor, image_size=(384, 384)):
    train_datasets = []
    val_datasets = []

    for dataset_name in os.listdir(dataset_root):
        dataset_path = os.path.join(dataset_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        train_image_dir = os.path.join(dataset_path, "train")
        train_mask_dir = os.path.join(dataset_path, "trainannot")
        val_image_dir = os.path.join(dataset_path, "val")
        val_mask_dir = os.path.join(dataset_path, "valannot")

        if os.path.exists(train_image_dir) and os.path.exists(train_mask_dir):
            train_datasets.append(SegmentationDataset(train_image_dir, train_mask_dir, processor, image_size))
        if os.path.exists(val_image_dir) and os.path.exists(val_mask_dir):
            val_datasets.append(SegmentationDataset(val_image_dir, val_mask_dir, processor, image_size))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    return train_dataset, val_dataset

# 加载模型和处理器
model_checkpoint = "facebook/mask2former-swin-base-coco-panoptic"
processor = AutoImageProcessor.from_pretrained(model_checkpoint)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_checkpoint)

# 加载数据
dataset_root = "./datasets/lane_labeled_expand_train_val"
train_dataset, val_dataset = load_datasets(dataset_root, processor)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    fp16=True,
    save_strategy="epoch",
    eval_strategy="steps",
    evaluation_strategy="steps",  # 兼容性
    eval_steps=200,
    logging_dir="./logs",
    logging_steps=50,
)

# 使用 Trainer 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,  # processor 将自动应用到数据
)

# 开始训练
trainer.train()