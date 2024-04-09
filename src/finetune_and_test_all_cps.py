import glob
import os
from datetime import datetime
from functools import partial
from itertools import repeat
from pathlib import Path

import pandas as pd

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger

from medimeta import MedIMeta, PickledMedIMetaTaskDataset
from torchcross.models.lightning import (
    SimpleClassifier,
    SimpleCrossDomainClassifier,
    CrossDomainMAML,
)
from torchvision import transforms

import torchcross as tx

from backbones import resnet18_backbone, resnet50_backbone

checkpoint_basedir = (
    "/mnt/qb/work/baumgartner/swoerner14/2023-mimeta/mimeta-experiments/experiments"
)


def main(args):
    training_types = args.training_type.split(",")
    checkpoints = ["imagenet_resnet18", "imagenet_resnet50"]
    checkpoints = [f"{checkpoint_basedir}/{cp}" for cp in checkpoints]

    for training_type in training_types:
        checkpoints += glob.glob(
            f"{checkpoint_basedir}/pretrained_2024-02/{args.target_dataset}/{training_type}*/**/best*.ckpt",
            recursive=True,
        )

    num_checkpoints = len(checkpoints)
    print(f"steps={args.finetuning_steps}_lr={args.learning_rate}")
    print(f"Number of checkpoints to test: {num_checkpoints}")
    print()
    print("Checkpoints:", checkpoints)

    for i, checkpoint in enumerate(checkpoints):
        print()
        print("-----------------------------------")
        print(f"Testing checkpoint {i}/{num_checkpoints}:")
        print(checkpoint)
        print("-----------------------------------")
        ft_and_test_cp(args, checkpoint)
        print("-----------------------------------")


def ft_and_test_cp(args, checkpoint_path):
    data_path = args.data_path
    presampled_data_path = args.presampled_data_path
    target_dataset_id = args.target_dataset
    target_task_name = args.target_task
    target_task_id = args.target_task_id

    if target_task_name and target_task_id:
        raise ValueError("Only one of target_task_name and target_task_id can be specified")
    elif target_task_name is None:
        dataset_info = MedIMeta.get_info_dict(data_path, target_dataset_id)
        target_task_name = dataset_info["tasks"][target_task_id]["task_name"]

    batch_size = 64

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

    # Create the test dataloader
    metatest_dataset = PickledMedIMetaTaskDataset(
        presampled_data_path,
        data_path,
        target_dataset_id,
        target_task_name,
        n_support=5,
        n_query=10,
        length=100,
        collate_fn=tx.utils.collate_fn.stack,
        transform=transforms.Compose(standard_transforms),
    )
    metatest_dataset.task_source.num_channels = 3

    hparams = {
        "lr": args.learning_rate,
    }

    # create unique experiment name and version
    now = datetime.now()
    save_dir = (
        f"{args.test_dir}/pretrained_2024-02/{target_dataset_id}_{target_task_name}/5shot"
        f"_steps={args.finetuning_steps}_lr={args.learning_rate}"
    )
    experiment_name = f"{os.path.relpath(checkpoint_path, checkpoint_basedir).replace('/', '_')}"
    version = now.strftime("%Y-%m-%d_%H-%M-%S")
    version_postfix = 0
    while Path(save_dir, experiment_name, f"{version}_{version_postfix}").exists():
        version_postfix += 1
    version = f"{version}_{version_postfix}"

    csv_logger = CSVLogger(
        save_dir, name=experiment_name, version=version, flush_logs_every_n_steps=1
    )

    # Create optimizer
    optimizer = partial(torch.optim.Adam, **hparams)

    if "fully_supervised" in checkpoint_path:
        checkpoint_model = SimpleClassifier
    elif "mmpft" in checkpoint_path:
        checkpoint_model = SimpleCrossDomainClassifier
    elif "mmmaml" in checkpoint_path:
        checkpoint_model = CrossDomainMAML
    elif "imagenet" in checkpoint_path:
        checkpoint_model = None
    else:
        raise ValueError("Unknown training type")

    if "resnet18" in checkpoint_path:
        backbone_fn = resnet18_backbone
    elif "resnet50" in checkpoint_path:
        backbone_fn = resnet50_backbone
    else:
        raise ValueError("Unknown backbone")

    if not os.path.exists(checkpoint_path):
        checkpoint_path = None

    if checkpoint_path is not None:
        cp = torch.load(checkpoint_path)
        bb_state_dict = {
            k[9:]: v for k, v in cp["state_dict"].items() if k.startswith("backbone.")
        }

    all_test_metrics = []

    for task in metatest_dataset:
        model = SimpleClassifier(
            *backbone_fn(pretrained=True),
            task.description,
            optimizer,
            expand_input_channels=False,
        )
        if checkpoint_path is not None:
            model.backbone.load_state_dict(bb_state_dict)

        # Create a new trainer for each task with the max_steps parameter
        trainer = pl.Trainer(
            max_steps=args.finetuning_steps,
            logger=[csv_logger],
            enable_checkpointing=False,
        )

        # Fine-tune the model on the current task
        trainer.fit(model, repeat(task.support), repeat(task.query, 1))

        test_metrics = trainer.test(model, repeat(task.query, 1))
        all_test_metrics.append(test_metrics[0])

    print("All Test Metrics:", all_test_metrics)
    df = pd.DataFrame(all_test_metrics)
    df.to_csv(f"{save_dir}/{experiment_name}/{version}_test_metrics.csv")

    # compute mean and std and count and save them to a csv with one row per mean/std/count
    mean = df.mean()
    std = df.std()
    count = df.count()
    agg_df = pd.concat([mean, std, count], axis=1)
    agg_df.columns = ["mean", "std", "count"]
    agg_df.to_csv(f"{save_dir}/{experiment_name}/{version}_test_metrics_aggregated.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument("--target_dataset", type=str, default="oct")
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--target_task_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--finetuning_steps", type=int, default=100)
    parser.add_argument("--presampled_data_path", type=str, default="data/MedIMeta_presampled")
    parser.add_argument("--training_type", type=str, default="mmpft")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--test_dir", type=str, default="./experiments/test")

    main(parser.parse_args())
