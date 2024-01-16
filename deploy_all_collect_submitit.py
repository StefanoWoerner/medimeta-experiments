import os
import submitit

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

submitit_logs_dir = "./experiments/cluster_logs"
executor = submitit.AutoExecutor(folder=submitit_logs_dir)

# SBATCH --ntasks=1                # Number of tasks (see below)
# SBATCH --cpus-per-task=8         # Number of CPU cores per task
# SBATCH --nodes=1                 # Ensure that all cores are on one machine
# SBATCH --time=2-00:00            # Runtime in D-HH:MM
# SBATCH --partition=2080-preemptable-galvani # Partition to submit to
# SBATCH --gres=gpu:1              # optionally type and number of gpus
# SBATCH --mem=40G                 # Memory pool for all cores (see also --mem-per-cpu)
# SBATCH --output=slurm_output/%j.out  # File to which STDOUT will be written
# SBATCH --error=slurm_output/%j.err   # File to which STDERR will be written
# SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
# SBATCH --mail-user=stefano.woerner@uni-tuebingen.de  # Email to which notifications will be sent

executor.update_parameters(
    timeout_min=60 * 48,
    slurm_partition="2080-preemptable-galvani",
    slurm_gres="gpu:1",
    slurm_mem="40G",
    slurm_nodes=1,
    slurm_ntasks=1,
    slurm_cpus_per_task=8,
    slurm_array_parallelism=100,
    slurm_additional_parameters={
        "constraint": "2080Ti",
        "account": "galvani",
        "qos": "galvani-preemptable",
    },
)

# learning_rates = [1e-3, 1e-2]
# finetuning_stepss = [50, 100]
learning_rates = []
finetuning_stepss = []

# training_type = "mmpft,mmmaml"
# training_type = "fully_supervised"

# test_method = "5shot"
test_method = "fulltest"

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
