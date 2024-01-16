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

collection_dir = "./experiments/test_selection/fulltest_selection"


def main():
    dfs = []
    for dataset in datasets:
        csv_paths = glob.glob(f"{collection_dir}/{dataset}*.csv")
        if len(csv_paths) == 0:
            continue
        csv_path = csv_paths[0]
        df = pd.read_csv(csv_path, index_col=0)
        df["Dataset"] = dataset
        dfs.append(df)

    result = pd.concat(dfs)
    # restructure the dataframe so that there is one row per backbone and one column per dataset
    result = result.pivot(index="Backbone", columns="Dataset", values="mean auroc")
    print(result)
    result.to_csv(f"{collection_dir}.csv")


if __name__ == "__main__":
    main()
