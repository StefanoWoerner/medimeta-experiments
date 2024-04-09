import os

datasets = [
    # "aml",
    # "bus",
    "mammo_mass",
    "mammo_calc",
    "cxr",
    "derm",
    "oct",
    "pneumonia",
    # "crc",
    # "pbc",
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
backbones = ["resnet18"]  # , "resnet50"]
learning_rates = [1e-3, 1e-4]

augmentations = [True]
alltaskss = [True, False]
scale_loss_by_batch_counts = [False]

class_weightings = [False]

for dataset in datasets:
    for backbone in backbones:
        for learning_rate in learning_rates:
            for augmentation in augmentations:
                for alltasks in alltaskss:
                    for scale_loss_by_batch_count in scale_loss_by_batch_counts:
                        for class_weighting in class_weightings:
                            command = (
                                f"sbatch /mnt/qb/work/baumgartner/swoerner14/2023-mimeta/mimeta-experiments/deploy.sh "
                                f"--data_path=data/MIMeta "
                                f"--target_dataset={dataset} "
                                f"--backbone={backbone} "
                                f"--learning_rate={learning_rate}"
                            )
                            if augmentation:
                                command += " --use_data_augmentation"
                            if alltasks:
                                command += " --all_tasks"
                            if scale_loss_by_batch_count:
                                command += " --scale_loss_by_batch_count"
                            if class_weighting:
                                command += " --use_class_weighting"
                            # submit job
                            os.system(command)
