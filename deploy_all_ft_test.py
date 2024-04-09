import os

datasets = [
    # "aml",
    # "bus",
    # "mammo_mass",
    # "mammo_calc",
    # "cxr",
    # "derm",
    # "oct",
    "pneumonia",
    # "crc",
    "pbc",
    # "fundus",
    # "dr_regular",
    # "dr_uwf",
    # "glaucoma",
    # "organs_axial",
    # "organs_coronal",
    # "organs_sagittal",
    # "skinl_clinic",
    # "skinl_derm",
]

learning_rates = [1e-3, 1e-2]
finetuning_stepss = [50, 100]

training_type = "mmmaml"  # "mmpft"  # ,mmmaml"

test_dir = "./experiments/test"

for dataset in datasets:
    for learning_rate in learning_rates:
        for finetuning_steps in finetuning_stepss:
            command = (
                f"sbatch deploy_ft_test.sh "
                f"--data_path=data/MIMeta "
                f"--presampled_data_path=data/MIMeta_presampled "
                f"--target_dataset={dataset} "
                f"--learning_rate={learning_rate} "
                f"--finetuning_steps={finetuning_steps} "
                f"--training_type={training_type} "
                f"--test_dir={test_dir}"
            )
            # submit job
            os.system(command)
