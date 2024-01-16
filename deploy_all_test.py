import os

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

training_type = "fully_supervised"

num_workers = 8

for dataset in datasets:
    command = (
        f"sbatch deploy_test.sh "
        f"--data_path=data/MIMeta "
        f"--target_dataset={dataset} "
        f"--num_workers={num_workers} "
        f"--training_type={training_type}"
    )
    # submit job
    os.system(command)
