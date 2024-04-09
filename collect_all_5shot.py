import glob
import pandas as pd

datasets = [
    "aml",
    "bus",
    "mammo_mass",
    "mammo_calc",
    "cxr",
    "derm",
    "oct",
    "pneumonia",
    "crc",
    "pbc",
    "fundus",
    "dr_regular",
    "dr_uwf",
    "glaucoma",
    "organs_axial",
    "organs_coronal",
    "organs_sagittal",
    "skinl_clinic",
    "skinl_derm",
]


finetuning_lrs = [1e-3, 1e-2]
finetuning_stepss = [50, 100]

collection_dir = "/mnt/qb/work/baumgartner/swoerner14/2023-mimeta/mimeta-experiments/experiments/test/pretrained_2024-02"
ending = "5shot_*_all_aggregated.csv"


def main():
    dfs = []
    for dataset in datasets:
        for finetuning_lr in finetuning_lrs:
            for finetuning_steps in finetuning_stepss:
                test_properties = f"_steps={finetuning_steps}_lr={finetuning_lr}"
                ending = f"5shot{test_properties}_all_aggregated.csv"
                csv_paths = glob.glob(f"{collection_dir}/{dataset}*{ending}")
                if len(csv_paths) == 0:
                    continue
                csv_path = csv_paths[0]
                df = pd.read_csv(csv_path, index_col=0)
                df["Dataset"] = dataset
                df["Fine-tuning Steps"] = finetuning_steps
                df["Fine-tuning Learning Rate"] = finetuning_lr
                conditions = (
                    (
                        (df["Backbone"] == "ResNet18")
                        & (
                            (df["Learning Rate"] == "0.001")
                            | (df["Learning Rate"] == "no training")
                        )
                    )
                    | (
                        (df["Backbone"] == "ResNet50")
                        & (
                            (df["Learning Rate"] == "0.0001")
                            | (df["Learning Rate"] == "no training")
                        )
                    )
                ) & (
                    (df["Data Augmentation"] == "no training")
                    | (df["Data Augmentation"] == "True")
                )
                df = df[conditions]
                dfs.append(df)

    # for dataset in datasets:
    #     csv_paths = glob.glob(f"{collection_dir_onlymain}/{dataset}*.csv")
    #     if len(csv_paths) == 0:
    #         continue
    #     csv_path = csv_paths[0]
    #     df = pd.read_csv(csv_path, index_col=0)
    #     df["Dataset"] = dataset
    #     df["pt tasks"] = "main"
    #     conditions = (
    #         (
    #             (df["Backbone"] == "ResNet18")
    #             & ((df["Learning Rate"] == "0.001") | (df["Learning Rate"] == "no training"))
    #         )
    #         | (
    #             (df["Backbone"] == "ResNet50")
    #             & ((df["Learning Rate"] == "0.0001") | (df["Learning Rate"] == "no training"))
    #         )
    #     ) & ((df["Data Augmentation"] == "no training") | (df["Data Augmentation"] == "True"))
    #     df = df[conditions]
    #     dfs.append(df)

    result = pd.concat(dfs)
    # restructure the dataframe so that there is one row per backbone and one column per dataset
    result = result.pivot(
        index=[
            "Fine-tuning Steps",
            "Fine-tuning Learning Rate",
            "Training Type",
            "Backbone",
            "Task Loss Scaling",
            "All Tasks",
            "Checkpoint Metric",
        ],
        columns=["Dataset"],
        values=["mean auroc", "ci95 auroc"],
    )
    print(result)
    result.to_pickle(f"{collection_dir}_5shot2.pkl")
    result.to_csv(f"{collection_dir}_5shot2.csv")


if __name__ == "__main__":
    main()
