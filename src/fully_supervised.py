from datetime import datetime
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from medimeta import MedIMeta
from torch.utils.data import DataLoader
from torchcross.data.task import TaskTarget
from torchcross.models.lightning import SimpleClassifier
from torchvision import transforms

from backbones import get_backbone


debug = False


def main(args):
    data_path = args.data_path
    target_dataset_id = args.target_dataset
    target_task_name = args.target_task
    target_task_id = args.target_task_id
    num_workers = args.num_workers
    batch_size = args.batch_size

    if target_task_name and target_task_id:
        raise ValueError("Only one of target_task_name and target_task_id can be specified")
    elif target_task_name is None:
        dataset_info = MedIMeta.get_info_dict(data_path, target_dataset_id)
        target_task_name = dataset_info["tasks"][target_task_id]["task_name"]

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

    # Create train and validation datasets for the target task
    train_dataset = MedIMeta(
        data_path,
        target_dataset_id,
        target_task_name,
        split="train",
        transform=transforms.Compose(augmentation_transforms + standard_transforms),
    )
    val_dataset = MedIMeta(
        data_path,
        target_dataset_id,
        target_task_name,
        split="val",
        transform=transforms.Compose(standard_transforms),
    )
    # test_dataset = MIMeta(
    #     data_path,
    #     target_dataset_id,
    #     target_task_name,
    #     split="test",
    #     transform=transforms.Compose(standard_transforms),
    # )
    train_dataset.num_channels = 3
    val_dataset.num_channels = 3
    # test_dataset.num_channels = 3

    if debug:
        print("Task description:")
        print(train_dataset.task_description)
        print(val_dataset.task_description)
        # print(test_dataset.task_description)

        print("Train dataset:")
        print(train_dataset)

        print("Val dataset:")
        print(val_dataset)

    # Create dataloaders for the target task
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=True,
    # )

    hparams = {
        "lr": args.learning_rate,
    }

    # Create optimizer
    optimizer = partial(torch.optim.Adam, **hparams)

    task_description = train_dataset.task_description

    backbone_name = args.backbone
    backbone, num_backbone_features = get_backbone(backbone_name)

    pos_class_weights = None
    if args.use_class_weighting:
        pos_class_weights = get_class_weights(train_dataset)

    # Create the lighting model with pre-trained resnet18 backbone
    model = SimpleClassifier(
        backbone,
        num_backbone_features,
        task_description,
        optimizer,
        expand_input_channels=False,
        pos_class_weights=pos_class_weights,
    )

    if debug:
        print(model)

    # create unique experiment name and version
    now = datetime.now()
    save_dir = f"./experiments/fullsup_{now.strftime('%Y-%m')}"
    experiment_name = (
        f"{target_dataset_id}_{target_task_name}/fully_supervised_{backbone_name}_"
        f"lr={args.learning_rate}_augmentation={args.use_data_augmentation}"
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
        # val_check_interval=0.5,
        # check_val_every_n_epoch=None,
        # val_check_interval=2000,
        # limit_val_batches=200,
        logger=[tb_logger, csv_logger],
        callbacks=checkpoint_callbacks + [early_stopping],
    )

    # Train the model on the target task
    trainer.fit(model, train_dataloader, val_dataloader)

    # get final validation and test metrics
    val_metrics = trainer.validate(model, val_dataloader)
    # test_metrics = trainer.test(model, test_dataloader)


def get_class_weights(train_dataset):
    task_description = train_dataset.task_description
    pos_samples_per_class = train_dataset.get_num_samples_per_class().astype(np.float32)
    neg_samples_per_class = train_dataset.get_num_samples_per_class(neg=True).astype(np.float32)
    match task_description.task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            pos_class_weights = torch.from_numpy(
                (pos_samples_per_class.max() + 1) / (pos_samples_per_class + 1)
            )
        case TaskTarget.MULTILABEL_CLASSIFICATION | TaskTarget.BINARY_CLASSIFICATION:
            pos_class_weights = torch.from_numpy(
                (neg_samples_per_class + 1) / (pos_samples_per_class + 1)
            )
        case TaskTarget.ORDINAL_REGRESSION:
            pos_class_weights = torch.from_numpy(
                (pos_samples_per_class.max() + 1) / (pos_samples_per_class + 1)
            )
        case target:
            raise NotImplementedError(f"Class weighting for task target {target} not implemented")
    return pos_class_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument("--target_dataset", type=str, default="aml")
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--target_task_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_data_augmentation", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--use_class_weighting", action="store_true")

    main(parser.parse_args())
