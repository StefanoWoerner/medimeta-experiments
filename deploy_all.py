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
backbones = ["resnet18", "resnet50"]

augmentations = [False, True]

learning_rates = [1e-3, 1e-4]

class_weightings = [False]

num_workers = 8

for dataset in datasets:
    for backbone in backbones:
        for augmentation in augmentations:
            for learning_rate in learning_rates:
                for class_weighting in class_weightings:
                    command = (
                        f"sbatch deploy.sh "
                        f"--data_path=data/MIMeta "
                        f"--target_dataset={dataset} "
                        f"--backbone={backbone} "
                        f"--learning_rate={learning_rate} "
                        f"--num_workers={num_workers}"
                    )
                    if augmentation:
                        command += " --use_data_augmentation"
                    if class_weighting:
                        command += " --use_class_weighting"
                    # submit job
                    os.system(command)
