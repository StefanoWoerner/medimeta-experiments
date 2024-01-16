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

collection_dir = "./experiments/test_selection/5shot_selection_2023"
collection_dir_onlymain = "./experiments/test_selection/5shot_selection_2024"


def main():
    dfs = []
    for dataset in datasets:
        csv_paths = glob.glob(f"{collection_dir}/{dataset}*.csv")
        if len(csv_paths) == 0:
            continue
        csv_path = csv_paths[0]
        df = pd.read_csv(csv_path, index_col=0)
        df["Dataset"] = dataset
        df["pt tasks"] = "all"
        conditions = (
            (
                (df["Backbone"] == "ResNet18")
                & ((df["Learning Rate"] == "0.001") | (df["Learning Rate"] == "no training"))
            )
            | (
                (df["Backbone"] == "ResNet50")
                & ((df["Learning Rate"] == "0.0001") | (df["Learning Rate"] == "no training"))
            )
        ) & ((df["Data Augmentation"] == "no training") | (df["Data Augmentation"] == "True"))
        df = df[conditions]
        dfs.append(df)

    for dataset in datasets:
        csv_paths = glob.glob(f"{collection_dir_onlymain}/{dataset}*.csv")
        if len(csv_paths) == 0:
            continue
        csv_path = csv_paths[0]
        df = pd.read_csv(csv_path, index_col=0)
        df["Dataset"] = dataset
        df["pt tasks"] = "main"
        conditions = (
            (
                (df["Backbone"] == "ResNet18")
                & ((df["Learning Rate"] == "0.001") | (df["Learning Rate"] == "no training"))
            )
            | (
                (df["Backbone"] == "ResNet50")
                & ((df["Learning Rate"] == "0.0001") | (df["Learning Rate"] == "no training"))
            )
        ) & ((df["Data Augmentation"] == "no training") | (df["Data Augmentation"] == "True"))
        df = df[conditions]
        dfs.append(df)

    result = pd.concat(dfs)
    # restructure the dataframe so that there is one row per backbone and one column per dataset
    result = result.pivot(
        index=["Training Type", "Backbone", "pt tasks"],
        columns=["Dataset"],
        values="mean auroc",
    )
    print(result)
    result.to_csv(f"{collection_dir}.csv")


if __name__ == "__main__":
    main()
