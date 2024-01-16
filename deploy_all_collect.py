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

learning_rates = [1e-3, 1e-2]
finetuning_stepss = [50, 100]
# learning_rates = []
# finetuning_stepss = []

# training_type = "mmpft,mmmaml"
# training_type = "fully_supervised"

# test_method = "5shot"
test_method = "5shot"

test_dir = "./experiments/test"
significant_digits = 3

for dataset in datasets:
    if len(learning_rates) > 0 and len(finetuning_stepss) > 0:
        for learning_rate in learning_rates:
            for finetuning_steps in finetuning_stepss:
                command = (
                    f"sbatch deploy_collect.sh "
                    f"--data_path=data/MIMeta "
                    f"--target_dataset={dataset} "
                    f"--learning_rate={learning_rate} "
                    f"--finetuning_steps={finetuning_steps} "
                    f"--test_method={test_method} "
                    f"--test_dir={test_dir} "
                    f"--significant_digits={significant_digits}"
                )
                # submit job
                os.system(command)
    else:
        command = (
            f"sbatch deploy_collect.sh "
            f"--data_path=data/MIMeta "
            f"--target_dataset={dataset} "
            f"--test_method={test_method} "
            f"--test_dir={test_dir} "
            f"--significant_digits={significant_digits}"
        )
        # submit job
        os.system(command)
