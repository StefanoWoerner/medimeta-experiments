from datetime import datetime
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
import torchopt
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

import torchcross as tx

from mimeta import MIMeta, get_available_tasks, MultiPickledMIMetaTaskDataset
from torchcross.models.lightning import CrossDomainMAML
from torchcross.utils.collate_fn import identity
from torchvision import transforms

from backbones import resnet18_backbone, resnet50_backbone


def main(args):
    data_path = args.data_path
    presampled_data_path = args.presampled_data_path
    target_dataset_id = args.target_dataset
    target_task_name = args.target_task
    target_task_id = args.target_task_id
    num_workers = args.num_workers

    batch_size = 1

    dataset_info = MIMeta.get_info_dict(data_path, target_dataset_id)
    if target_task_name and target_task_id:
        raise ValueError(
            "Only one of target_task_name and target_task_id can be specified"
        )
    elif target_task_name is None:
        target_task_name = dataset_info["tasks"][target_task_id]["task_name"]

    all_overlaps = dataset_info["domain_overlaps"] + dataset_info["subject_overlaps"]

    task_dict = get_available_tasks(data_path)
    train_val_tasks = [
        (ds, t)
        for ds, tasks in task_dict.items()
        for t in tasks[:1]
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
            transforms.RandomRotation(
                10, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomApply([random_crop_and_resize], p=0.25),
        ]
    standard_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # Create the cross-domain meta-dataset from pre-sampled tasks
    train_dataset = MultiPickledMIMetaTaskDataset(
        presampled_data_path,
        data_path,
        train_val_tasks,
        n_support=(1, 10),
        n_query=10,
        length=1000,
        collate_fn=tx.utils.collate_fn.stack,
        transform=transforms.Compose(augmentation_transforms + standard_transforms),
        # split="train",
    )
    val_dataset = MultiPickledMIMetaTaskDataset(
        presampled_data_path,
        data_path,
        train_val_tasks,
        n_support=(1, 10),
        n_query=10,
        length=100,
        collate_fn=tx.utils.collate_fn.stack,
        transform=transforms.Compose(standard_transforms),
        # split="val",
    )

    for dataset in train_dataset.task_sources:
        dataset.num_channels = 3
    for dataset in val_dataset.task_sources:
        dataset.num_channels = 3

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=True,
    )

    outer_optim_hparams = {
        "lr": args.learning_rate,
    }

    # Create optimizer
    outer_optimizer = partial(torch.optim.Adam, **outer_optim_hparams)
    inner_optimizer = partial(torchopt.MetaSGD, lr=0.1)
    eval_inner_optimizer = partial(torch.optim.SGD, lr=0.1)
    num_inner_steps = 4
    eval_num_inner_steps = 32

    backbone_name = args.backbone
    if backbone_name == "resnet18":
        backbone = resnet18_backbone(pretrained=True)
    elif backbone_name == "resnet50":
        backbone = resnet50_backbone(pretrained=True)
    else:
        raise ValueError(f"Unknown backbone {backbone_name}")

    # Create the lighting model with pre-trained resnet18 backbone
    model = CrossDomainMAML(
        *backbone,
        outer_optimizer,
        inner_optimizer,
        eval_inner_optimizer,
        num_inner_steps,
        eval_num_inner_steps,
    )

    # create unique experiment name and version
    now = datetime.now()
    save_dir = f"./experiments/{now.strftime('%Y-%m-%d')}"
    experiment_name = (
        f"{target_dataset_id}_{target_task_name}_mmmaml_{backbone_name}_"
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
            monitor="loss/metaval/query",
            filename="best-loss-{epoch}-{step}",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="AUROC/metaval/query",
            filename="best-AUROC-{epoch}-{step}",
            mode="max",
            save_top_k=1,
        ),
    ]
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="AUROC/metaval/query", patience=10, mode="max"
    )

    # Create the lightning trainer
    trainer = pl.Trainer(
        max_steps=500_000,
        # val_check_interval=0.5,
        check_val_every_n_epoch=None,
        val_check_interval=2000,
        limit_val_batches=200,
        logger=[tb_logger, csv_logger],
        callbacks=checkpoint_callbacks + [early_stopping],
    )

    # Train the model on the target task
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MIMeta")
    parser.add_argument(
        "--presampled_data_path", type=str, default="data/MIMeta_presampled"
    )
    parser.add_argument("--target_dataset", type=str, default="oct")
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--target_task_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="resnet18")

    main(parser.parse_args())
