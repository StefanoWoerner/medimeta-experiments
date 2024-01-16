from mimeta import MIMeta

data_path = "data/MIMeta"
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

splits = ["train", "val", "test"]


def main():
    for target_dataset_id in datasets:
        dataset_info = MIMeta.get_info_dict(data_path, target_dataset_id)
        for task_info in dataset_info["tasks"]:
            target_task_name = task_info["task_name"]
            for s in splits:
                dataset = MIMeta(
                    data_path,
                    target_dataset_id,
                    target_task_name,
                    split=s,
                )
                ns = dataset.get_num_samples_per_class()
                print(ns)
                if any(n < 1 for n in ns):
                    print(
                        f"{target_dataset_id} {target_task_name} {s} split has a class with no positive samples"
                    )
                    # exit(1)


if __name__ == "__main__":
    main()
