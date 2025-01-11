#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

import albumentations as A
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Image, load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import transformers
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor

class DatasetLoader:
    """数据集加载器类"""
    
    @staticmethod
    def load_single_dataset(dataset_path: str) -> DatasetDict:
        """加载单个数据集"""
        def load_images_and_annotations(image_dir, annot_dir):
            image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                                if f.endswith(('jpg', 'png', 'bmp'))])
            annotation_paths = sorted([os.path.join(annot_dir, f) for f in os.listdir(annot_dir) 
                                    if f.endswith(('jpg', 'png', 'bmp'))])
            
            assert len(image_paths) == len(annotation_paths), "图像和标注数量不匹配！"
            for img, ann in zip(image_paths, annotation_paths):
                assert os.path.splitext(os.path.basename(img))[0] == os.path.splitext(os.path.basename(ann))[0], \
                    f"图像和标注文件名不匹配: {img} vs {ann}"

            dataset = Dataset.from_dict({"image": image_paths, "annotation": annotation_paths})
            dataset = dataset.cast_column("image", Image())
            dataset = dataset.cast_column("annotation", Image())
            return dataset

        train_dataset = load_images_and_annotations(
            os.path.join(dataset_path, "train"),
            os.path.join(dataset_path, "trainannot"),
        )
        val_dataset = load_images_and_annotations(
            os.path.join(dataset_path, "val"),
            os.path.join(dataset_path, "valannot"),
        )

        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
        })

    @staticmethod
    def load_all_datasets(root_dir: str) -> DatasetDict:
        """加载多个数据集并合并"""
        all_train_datasets = []
        all_val_datasets = []

        for dataset_name in os.listdir(root_dir):
            dataset_path = os.path.join(root_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            print(f"正在加载数据集: {dataset_name}")
            dataset = DatasetLoader.load_single_dataset(dataset_path)
            all_train_datasets.append(dataset["train"])
            all_val_datasets.append(dataset["validation"])

        from datasets import concatenate_datasets
        combined_train = concatenate_datasets(all_train_datasets) if len(all_train_datasets) > 1 else all_train_datasets[0]
        combined_val = concatenate_datasets(all_val_datasets) if len(all_val_datasets) > 1 else all_val_datasets[0]

        return DatasetDict({
            "train": combined_train,
            "validation": combined_val,
        })

class DataProcessor:
    """数据预处理器类"""
    
    @staticmethod
    def preprocess_masks(batch):
        """预处理掩码标签"""
        processed_batch = {
            "image": batch["image"],
            "annotation": np.array(batch["annotation"])
        }
        annotation = processed_batch["annotation"]
        
        annotation[annotation == 255] = 255
        annotation = np.where(annotation > 127, 1, 0)
        annotation = annotation.astype(np.int64)
        processed_batch["annotation"] = annotation
        return processed_batch

    @staticmethod
    def augment_and_transform_batch(
        examples: Mapping[str, Any], 
        transform: A.Compose, 
        image_processor: AutoImageProcessor
    ) -> BatchFeature:
        """数据增强和转换"""
        batch = {
            "pixel_values": [],
            "mask_labels": [],
            "class_labels": [],
        }
        
        global_class_labels = set()

        for pil_image, pil_annotation in zip(examples["image"], examples["annotation"]):
            image = np.array(pil_image)
            semantic_and_instance_masks = np.array(pil_annotation.convert("L"))

            if semantic_and_instance_masks.ndim == 1:
                height, width = pil_annotation.size
                semantic_and_instance_masks = semantic_and_instance_masks.reshape((height, width))

            unique_classes = np.unique(semantic_and_instance_masks)
            global_class_labels.update(unique_classes)

            output = transform(image=image, mask=semantic_and_instance_masks)
            aug_image = output["image"]
            aug_instance_mask = output["mask"]

            unique_semantic_ids = np.unique(aug_instance_mask)
            instance_id_to_semantic_id = {instance_id: instance_id for instance_id in unique_semantic_ids}

            model_inputs = image_processor(
                images=[aug_image],
                segmentation_maps=[aug_instance_mask],
                instance_id_to_semantic_id=instance_id_to_semantic_id,
                return_tensors="pt",
            )

            batch["pixel_values"].append(model_inputs.pixel_values[0])
            batch["mask_labels"].append(model_inputs.mask_labels[0])
            batch["class_labels"].append(model_inputs.class_labels[0])

        return batch

class Evaluator:
    """评估器类"""
    
    def __init__(
        self,
        image_processor: AutoImageProcessor,
        id2label: Mapping[int, str],
        threshold: float = 0.0,
    ):
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = self.get_metric()

    def get_metric(self):
        return MeanAveragePrecision(iou_type="segm", class_metrics=True)

    def reset_metric(self):
        self.metric.reset()

    def postprocess_target_batch(self, target_batch) -> List[Dict[str, torch.Tensor]]:
        batch_masks = target_batch[0]
        batch_labels = target_batch[1]
        post_processed_targets = []
        for masks, labels in zip(batch_masks, batch_labels):
            post_processed_targets.append({
                "masks": masks.to(dtype=torch.bool),
                "labels": labels,
            })
        return post_processed_targets

    def get_target_sizes(self, post_processed_targets) -> List[List[int]]:
        return [target["masks"].shape[-2:] for target in post_processed_targets]

    def postprocess_prediction_batch(self, prediction_batch, target_sizes) -> List[Dict[str, torch.Tensor]]:
        model_output = ModelOutput(
            class_queries_logits=prediction_batch[0], 
            masks_queries_logits=prediction_batch[1]
        )
        post_processed_output = self.image_processor.post_process_instance_segmentation(
            model_output,
            threshold=self.threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        post_processed_predictions = []
        for image_predictions, target_size in zip(post_processed_output, target_sizes):
            if image_predictions["segments_info"]:
                post_processed_image_prediction = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                    "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]]),
                    "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]]),
                }
            else:
                post_processed_image_prediction = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                    "labels": torch.tensor([]),
                    "scores": torch.tensor([]),
                }
            post_processed_predictions.append(post_processed_image_prediction)

        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results: EvalPrediction, compute_result: bool = False) -> Mapping[str, float]:
        prediction_batch = nested_cpu(evaluation_results.predictions)
        target_batch = nested_cpu(evaluation_results.label_ids)

        post_processed_targets = self.postprocess_target_batch(target_batch)
        target_sizes = self.get_target_sizes(post_processed_targets)
        post_processed_predictions = self.postprocess_prediction_batch(prediction_batch, target_sizes)

        self.metric.update(post_processed_predictions, post_processed_targets)

        if not compute_result:
            return

        metrics = self.metric.compute()
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
        self.reset_metric()
        return metrics

class TrainingManager:
    """训练管理器类"""
    
    @staticmethod
    def setup_logging(training_args: TrainingArguments) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if training_args.should_log:
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    @staticmethod
    def find_last_checkpoint(training_args: TrainingArguments) -> Optional[str]:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            checkpoint = get_last_checkpoint(training_args.output_dir)
            if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        return checkpoint

def collate_fn(examples):
    """数据批处理函数"""
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    return batch

def nested_cpu(tensors):
    """将张量递归地移到CPU上"""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors

def main():
    # 配置参数
    config = {
        'model': {
            'name': 'facebook/mask2former-swin-tiny-coco-instance',
            'num_labels': 2,
            'image_height': 512,
            'image_width': 512,
        },
        'training': {
            'output_dir': './output',
            'num_train_epochs': 100,
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'warmup_ratio': 0.1,
            'logging_steps': 10,
            'evaluation_strategy': 'steps',
            'eval_steps': 100,
            'save_strategy': 'steps',
            'save_steps': 100,
            'learning_rate': 5e-5,
            'do_train': True,
            'do_eval': True,
            'remove_unused_columns': False,  # 添加这个配置
            'report_to': 'tensorboard',
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'eval_do_concat_batches': False,
            'batch_eval_metrics': True,
        },
        'data': {
            'dataset_path': 'datasets/lane_labeled_expand_train_val',
            'label2id': {"background": 0, "lane": 1},
        }
    }

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 创建训练参数
    training_args = TrainingArguments(
        **config['training']
    )


    # 设置日志
    TrainingManager.setup_logging(training_args)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: "
        f"{training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 加载数据集
    dataset = DatasetLoader.load_all_datasets(config['data']['dataset_path'])
    dataset = dataset.map(DataProcessor.preprocess_masks, batched=False, num_proc=8)

    # 加载模型和图像处理器
    id2label = {v: k for k, v in config['data']['label2id'].items()}
    
    model = AutoModelForUniversalSegmentation.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels'],
        label2id=config['data']['label2id'],
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        config['model']['name'],
        do_resize=True,
        size={
            "height": config['model']['image_height'], 
            "width": config['model']['image_width']
        },
    )

    # 定义数据增强
    train_augment_and_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
        A.Resize(
            height=config['model']['image_height'], 
            width=config['model']['image_width'], 
            always_apply=True
        ),
    ])
    
    validation_transform = A.Compose([
        A.Resize(
            height=config['model']['image_height'], 
            width=config['model']['image_width'], 
            always_apply=True
        ),
    ])

    # 定义数据转换函数
    train_transform_batch = partial(
        DataProcessor.augment_and_transform_batch,
        transform=train_augment_and_transform,
        image_processor=image_processor
    )
    validation_transform_batch = partial(
        DataProcessor.augment_and_transform_batch,
        transform=validation_transform,
        image_processor=image_processor
    )

    # 应用数据转换
    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)

    # 创建评估器
    compute_metrics = Evaluator(
        image_processor=image_processor,
        id2label=id2label,
        threshold=0.0
    )

    # 查找最后的检查点
    checkpoint = TrainingManager.find_last_checkpoint(training_args)

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # 训练
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 评估
    if training_args.do_eval:
        metrics = trainer.evaluate(
            eval_dataset=dataset["validation"],
            metric_key_prefix="test"
        )
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

if __name__ == "__main__":
    main()