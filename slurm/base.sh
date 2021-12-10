#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --job-name=distill
#SBATCH --gres=gpu:4
#SBATCH -o /pfs/work7/workspace/scratch/ma_mkuc-clkd_data/slurm/outputs/slurm-%j.out
#SBATCH --mail-user=mkuc@mail.uni-mannheim.de
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

now=$(date +"%T")
echo "Program starts:  $now"

# Activate conda env
# srun $1
# Run script
srun python /home/ma/ma_ma/$SLURM_JOB_USER/clkd/run.py

end=$(date +"%T")
echo "Completed: $end"