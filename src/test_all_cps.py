import glob
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger

from medimeta import MedIMeta
from torch.utils.data import DataLoader
from torchcross.models.lightning import (
    SimpleClassifier,
    SimpleCrossDomainClassifier,
    CrossDomainMAML,
)
from torchvision import transforms


from backbones import resnet18_backbone, resnet50_backbone

checkpoint_basedir = (
    "/mnt/qb/work/baumgartner/swoerner14/2023-mimeta/mimeta-experiments/experiments"
)


def main(args):
    training_types = args.training_type.split(",")
    checkpoints = ["imagenet_resnet18", "imagenet_resnet50"]
    checkpoints = [f"{checkpoint_basedir}/{cp}" for cp in checkpoints]

    target_task_name = args.target_task
    target_task_id = args.target_task_id

    if target_task_name and target_task_id:
        raise ValueError("Only one of target_task_name and target_task_id can be specified")
    elif target_task_name is None:
        dataset_info = MedIMeta.get_info_dict(args.data_path, args.target_dataset)
        target_task_name = dataset_info["tasks"][target_task_id]["task_name"]

    for training_type in training_types:
        checkpoints += glob.glob(
            f"{checkpoint_basedir}/fullsup_2024*/{args.target_dataset}_{target_task_name}/{training_type}*/**/best*.ckpt",
            recursive=True,
        )

    print("Dataset:", args.target_dataset)
    print("Training Types:", training_types)
    print("Checkpoints:", checkpoints)

    for checkpoint in checkpoints:
        print("Testing checkpoint:", checkpoint)
        test_cp(args, checkpoint)


def test_cp(args, checkpoint_path):
    data_path = args.data_path
    target_dataset_id = args.target_dataset
    target_task_name = args.target_task
    target_task_id = args.target_task_id
    num_workers = args.num_workers

    if target_task_name and target_task_id:
        raise ValueError("Only one of target_task_name and target_task_id can be specified")
    elif target_task_name is None:
        dataset_info = MedIMeta.get_info_dict(data_path, target_dataset_id)
        target_task_name = dataset_info["tasks"][target_task_id]["task_name"]

    batch_size = 64

    standard_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # Create the val dataloader
    val_dataset = MedIMeta(
        data_path,
        target_dataset_id,
        target_task_name,
        split="val",
        transform=transforms.Compose(standard_transforms),
    )
    val_dataset.num_channels = 3

    # Create the test dataloader
    test_dataset = MedIMeta(
        data_path,
        target_dataset_id,
        target_task_name,
        split="test",
        transform=transforms.Compose(standard_transforms),
    )
    test_dataset.num_channels = 3

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    hparams = {
        "lr": 1e-3,
    }

    # create unique experiment name and version
    now = datetime.now()
    save_dir = (
        f"./experiments/test/fullsup_2024-02/{target_dataset_id}_{target_task_name}_fulltest"
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
        head_state_dict = {k[5:]: v for k, v in cp["state_dict"].items() if k.startswith("head.")}

    all_test_metrics = []

    model = SimpleClassifier(
        *backbone_fn(pretrained=True),
        test_dataset.task_description,
        optimizer,
        expand_input_channels=False,
    )
    if checkpoint_path is not None:
        model.backbone.load_state_dict(bb_state_dict)
        model.head.load_state_dict(head_state_dict)

    # Create a new trainer for each task with the max_steps parameter
    trainer = pl.Trainer(
        logger=[csv_logger],
        enable_checkpointing=False,
    )

    # get test metrics and val metrics
    val_metrics = trainer.validate(model, val_dataloader)
    test_metrics = trainer.test(model, test_dataloader)

    print("All Test Metrics:", test_metrics)
    df = pd.DataFrame(test_metrics)
    df_val = pd.DataFrame(val_metrics)
    df = pd.concat([df, df_val], axis=1)
    df.to_csv(f"{save_dir}/{experiment_name}/{version}_test_metrics.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument("--target_dataset", type=str, default="aml")
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--target_task_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--training_type", type=str, default="fully_supervised")
    parser.add_argument("--backbone", type=str, default="resnet18")

    main(parser.parse_args())
