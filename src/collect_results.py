import glob

import pandas as pd


from medimeta import MedIMeta


def main(args):
    target_dataset_id = args.target_dataset
    target_task_name = args.target_task
    target_task_id = args.target_task_id
    test_method = args.test_method
    test_dir = args.test_dir
    data_path = args.data_path
    significant_digits = args.significant_digits

    if target_task_name and target_task_id:
        raise ValueError("Only one of target_task_name and target_task_id can be specified")
    elif target_task_name is None:
        dataset_info = MedIMeta.get_info_dict(data_path, target_dataset_id)
        target_task_name = dataset_info["tasks"][target_task_id]["task_name"]

    if args.finetuning_steps is not None or args.learning_rate is not None:
        test_properties = f"_steps={args.finetuning_steps}_lr={args.learning_rate}"
    else:
        test_properties = ""

    metrics_dirs = glob.glob(
        f"{test_dir}/{target_dataset_id}_{target_task_name}/{test_method}{test_properties}/*"
    )

    print("Metrics Dirs:", metrics_dirs)
    print("Metric dirs count:", len(metrics_dirs))
    metrics_files = [sorted(glob.glob(f"{d}/*_test_metrics.csv"))[-1] for d in metrics_dirs]
    print("Metrics Files:", metrics_files)
    print("Metric files count:", len(metrics_files))
    assert len(metrics_files) == len(metrics_dirs)

    all_dfs = [pd.read_csv(f, index_col=0) for f in metrics_files]

    def aggregate_df(df):
        df_agg = pd.concat([df.mean(), df.std(), df.count()], axis=1)
        df_agg.columns = ["mean", "std", "count"]
        df_agg["sem"] = df_agg["std"] / df_agg["count"] ** 0.5
        df_agg["ci95"] = df_agg["sem"] * 1.96
        df_flattened = df_agg.stack().to_frame().T
        df_flattened.columns = [
            " ".join(map(str, col)).strip() for col in df_flattened.columns.values
        ]
        return df_flattened

    all_aggregated = [aggregate_df(df) for df in all_dfs]
    all_aggregated_df = pd.concat(all_aggregated)

    # use experiment name as index
    all_aggregated_df.index = [d.split("/")[-1] for d in metrics_dirs]

    def get_training_type(x):
        if "fully_supervised" in x:
            return "Fully Supervised"
        elif "mmpft" in x:
            return "MM-PFT"
        elif "mmmaml" in x:
            return "MM-MAML"
        elif "imagenet" in x:
            return "ImageNet"
        else:
            raise ValueError("Unknown training type")

    def get_backbone(x):
        if "resnet18" in x:
            return "ResNet18"
        elif "resnet50" in x:
            return "ResNet50"
        else:
            raise ValueError("Unknown backbone")

    def get_lr(x):
        lr_chunks = [chunk[3:] for chunk in x.split("_") if chunk.startswith("lr=")]
        if len(lr_chunks) == 0:
            return "no training"
        elif len(lr_chunks) > 1:
            raise ValueError("More than one lr chunk")
        else:
            return lr_chunks[0]

    def get_augmentation(x):
        aug_chunks = [chunk[13:] for chunk in x.split("_") if chunk.startswith("augmentation=")]
        if len(aug_chunks) == 0:
            return "no training"
        elif len(aug_chunks) > 1:
            raise ValueError("More than one augmentation chunk")
        else:
            return aug_chunks[0]

    def get_loss_scaling(x):
        loss_scaling_chunks = [
            chunk[16:] for chunk in x.split("_") if chunk.startswith("scale-task-loss=")
        ]
        if len(loss_scaling_chunks) == 0:
            return "not applicable"
        elif len(loss_scaling_chunks) > 1:
            raise ValueError("More than one loss scaling chunk")
        else:
            return loss_scaling_chunks[0]

    def get_alltasks(x):
        alltasks_chunks = [chunk[9:] for chunk in x.split("_") if chunk.startswith("alltasks=")]
        if len(alltasks_chunks) == 0:
            return "not applicable"
        elif len(alltasks_chunks) > 1:
            raise ValueError("More than one alltasks chunk")
        else:
            return alltasks_chunks[0]

    def get_checkpoint_metric_and_step(x):
        checkpoint_name_chunk = x.split("_")[-1]
        parts = checkpoint_name_chunk.split("-")
        if checkpoint_name_chunk.startswith("best-"):
            metric = "-".join(parts[:2])
        elif checkpoint_name_chunk.startswith("last"):
            metric = "last"
        else:
            metric = None
        step = parts[-1].split("=")[-1].split(".")[0] if parts[-1].startswith("step") else None
        return metric, step

    all_aggregated_df["Training Type"] = all_aggregated_df.index.map(get_training_type)
    all_aggregated_df["Backbone"] = all_aggregated_df.index.map(get_backbone)
    all_aggregated_df["Learning Rate"] = all_aggregated_df.index.map(get_lr)
    all_aggregated_df["Data Augmentation"] = all_aggregated_df.index.map(get_augmentation)
    all_aggregated_df["Task Loss Scaling"] = all_aggregated_df.index.map(get_loss_scaling)
    all_aggregated_df["All Tasks"] = all_aggregated_df.index.map(get_alltasks)
    all_aggregated_df["Checkpoint Metric"], all_aggregated_df["Checkpoint Step"] = zip(
        *all_aggregated_df.index.map(get_checkpoint_metric_and_step)
    )

    csv_save_path = f"{test_dir}/{target_dataset_id}_{target_task_name}_{test_method}{test_properties}_all_aggregated_raw.csv"
    all_aggregated_df.to_csv(csv_save_path)

    all_aggregated_df["mean auroc"] = all_aggregated_df["AUROC/test mean"].round(
        significant_digits
    )
    all_aggregated_df["mean accuracy"] = all_aggregated_df["accuracy/test mean"].round(
        significant_digits
    )
    all_aggregated_df["mean loss"] = all_aggregated_df["loss/test mean"].round(significant_digits)

    if "AUROC/test ci95" in all_aggregated_df.columns:
        all_aggregated_df["ci95 auroc"] = all_aggregated_df["AUROC/test ci95"].round(
            significant_digits
        )
        all_aggregated_df["ci95 accuracy"] = all_aggregated_df["accuracy/test ci95"].round(
            significant_digits
        )
        all_aggregated_df["ci95 loss"] = all_aggregated_df["loss/test ci95"].round(
            significant_digits
        )

    new_col_order = [
        "Training Type",
        "Backbone",
        "Learning Rate",
        "Data Augmentation",
        "Task Loss Scaling",
        "All Tasks",
        "Checkpoint Metric",
        "Checkpoint Step",
        "mean auroc",
        "ci95 auroc",
        "mean accuracy",
        "ci95 accuracy",
        "mean loss",
        "ci95 loss",
    ]

    if "AUROC/test ci95" not in all_aggregated_df.columns:
        new_col_order = [col for col in new_col_order if "ci95" not in col]

    all_aggregated_df = all_aggregated_df[new_col_order]

    csv_save_path = f"{test_dir}/{target_dataset_id}_{target_task_name}_{test_method}{test_properties}_all_aggregated.csv"
    all_aggregated_df.to_csv(csv_save_path)
    print("Saved to:", csv_save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument("--target_dataset", type=str, default="aml")
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--target_task_id", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--finetuning_steps", type=int, default=None)
    parser.add_argument("--test_method", type=str, default="5shot")
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/mnt/qb/work/baumgartner/swoerner14/2023-mimeta/mimeta-experiments/experiments/test/pretrained_2024-02",
    )
    parser.add_argument("--significant_digits", type=int, default=3)

    main(parser.parse_args())
