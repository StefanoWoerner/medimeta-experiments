import glob
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

args = parser.parse_args()

for dataset in datasets:
    checkpoints = glob.glob(f"experiments/20**{dataset}_{args.training_type}**/*.ckpt")
    for learning_rate in learning_rates:
        for finetuning_steps in finetuning_stepss:
            for checkpoint in checkpoints:
                command = (
                    f"python src/finetune_and_test.py "
                    f"--data_path=data/MIMeta "
                    f"--presampled_data_path=data/MIMeta_presampled "
                    f"--target_dataset={dataset} "
                    f"--learning_rate={learning_rate} "
                    f"--finetuning_steps={finetuning_steps} "
                    f"--checkpoint_path={checkpoint}"
                )
                # submit job
                os.system(command)
