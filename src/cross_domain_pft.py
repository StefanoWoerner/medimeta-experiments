from datetime import datetime
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

import torchcross as tx

from medimeta import MedIMeta, MultiMedIMetaBatchTaskSource, get_available_tasks
from torchcross.models.lightning import SimpleCrossDomainClassifier
from torchcross.utils.collate_fn import identity
from torchvision import transforms

from backbones import get_backbone


def main(args):
    data_path = args.data_path
    target_dataset_id = args.target_dataset
    target_task_name = args.target_task
    target_task_id = args.target_task_id
    num_workers = args.num_workers
    batch_size = args.batch_size

    dataset_info = MedIMeta.get_info_dict(data_path, target_dataset_id)
    if target_task_name and target_task_id:
        raise ValueError("Only one of target_task_name and target_task_id can be specified")
    elif target_task_name is None:
        target_task_name = dataset_info["tasks"][target_task_id]["task_name"]

    all_overlaps = dataset_info["domain_overlaps"] + dataset_info["subject_overlaps"]

    task_dict = get_available_tasks(data_path)
    if args.all_tasks:
        train_val_tasks = [
            (ds, t)
            for ds, tasks in task_dict.items()
            for t in tasks
            if ds != target_dataset_id and ds not in all_overlaps
        ]
    else:
        train_val_tasks = [
            (ds, tasks[0])
            for ds, tasks in task_dict.items()
            if ds != target_dataset_id and ds not in all_overlaps
        ]

    augmentation_transforms = []
    if args.use_data_augmentation:
        random_crop_and_resize = transforms.Compose(
            [
                transforms.RandomCrop(200),
                transforms.Resize(224),
            ]
        )

        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomApply([random_crop_and_resize], p=0.25),
        ]
    standard_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # Create the cross-domain batch task sources
    train_dataset = MultiMedIMetaBatchTaskSource(
        data_path,
        train_val_tasks,
        batch_size,
        collate_fn=tx.utils.collate_fn.stack,
        splits="train",
        transform=transforms.Compose(augmentation_transforms + standard_transforms),
    )
    val_dataset = MultiMedIMetaBatchTaskSource(
        data_path,
        train_val_tasks,
        batch_size,
        collate_fn=tx.utils.collate_fn.stack,
        splits="val",
        transform=transforms.Compose(augmentation_transforms + standard_transforms),
    )

    for dataset in train_dataset.datasets:
        dataset.task_source.num_channels = 3
    for dataset in val_dataset.datasets:
        dataset.task_source.num_channels = 3

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )

    hparams = {
        "lr": args.learning_rate,
    }

    # Create optimizer
    optimizer = partial(torch.optim.Adam, **hparams)

    backbone_name = args.backbone
    backbone, num_backbone_features = get_backbone(backbone_name)

    # Create the lighting model with pre-trained resnet18 backbone
    model = SimpleCrossDomainClassifier(
        backbone,
        num_backbone_features,
        optimizer,
        scale_loss_by_batch_count=args.scale_loss_by_batch_count,
    )

    # create unique experiment name and version
    now = datetime.now()
    save_dir = f"./experiments/pretrained_{now.strftime('%Y-%m')}"
    experiment_name = (
        f"{target_dataset_id}/mmpft_{backbone_name}_"
        f"lr={args.learning_rate}_augmentation={args.use_data_augmentation}_"
        f"scale-task-loss={args.scale_loss_by_batch_count}_"
        f"alltasks={args.all_tasks}"
    )
    version = now.strftime("%Y-%m-%d_%H-%M-%S")
    version_postfix = 0
    while Path(save_dir, experiment_name, f"{version}_{version_postfix}").exists():
        version_postfix += 1
    version = f"{version}_{version_postfix}"

    tb_logger = TensorBoardLogger(
        save_dir, name=experiment_name, version=version, default_hp_metric=False
    )
    csv_logger = CSVLogger(save_dir, name=experiment_name, version=version)

    checkpoint_callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="loss/val",
            filename="best-loss-{epoch}-{step}",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="AUROC/val",
            filename="best-AUROC-{epoch}-{step}",
            mode="max",
            save_top_k=1,
        ),
    ]
    early_stopping = pl.callbacks.EarlyStopping(monitor="AUROC/val", patience=10, mode="max")

    # Create the lightning trainer
    trainer = pl.Trainer(
        # max_steps=500_000,
        # val_check_interval=0.1,
        # accelerator="cpu",
        # check_val_every_n_epoch=None,
        # val_check_interval=2000,
        # limit_val_batches=10,
        logger=[tb_logger, csv_logger],
        callbacks=checkpoint_callbacks + [early_stopping],
    )

    # Train the model on the target task
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument("--target_dataset", type=str, default="oct")
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--target_task_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--scale_loss_by_batch_count", action="store_true")
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="resnet18")

    main(parser.parse_args())
