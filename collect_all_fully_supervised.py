import glob
import os

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

collection_dir = "./experiments/test/fullsup_2024-02"
ending = "fulltest_all_aggregated.csv"


def main():
    dfs = []
    for dataset in sorted(datasets):
        csv_paths = glob.glob(f"{collection_dir}/{dataset}*{ending}")
        for csv_path in csv_paths[:1]:
            df = pd.read_csv(csv_path, index_col=0)
            df["Dataset"] = dataset
            df["Task Name"] = os.path.basename(csv_path)[len(dataset) + 1 : -len(ending) - 1]
            conditions = (
                (
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
                )
                & (
                    (df["Data Augmentation"] == "no training")
                    | (df["Data Augmentation"] == "True")
                )
                # & ((df["Checkpoint Metric"] == "best-AUROC") | (df["Checkpoint Metric"] == np.nan))
            )

            df = df[conditions]
            dfs.append(df)

    result = pd.concat(dfs)
    # restructure the dataframe so that there is one row per backbone and one column per dataset
    result = result.pivot(
        index=["Backbone", "Checkpoint Metric"],
        columns=["Dataset", "Task Name"],
        values="mean auroc",
    )
    print(result)
    result.to_pickle(f"{collection_dir}.pkl")
    result.to_csv(f"{collection_dir}.csv")


if __name__ == "__main__":
    main()
