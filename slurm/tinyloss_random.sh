#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=tinyloss_random
#SBATCH --gres=gpu:8
#SBATCH --mem=192000
#SBATCH --time=35:00:00
#SBATCH -o /pfs/work7/workspace/scratch/ma_mkuc-clkd_data/slurm/outputs/slurm-%j.out
#SBATCH --mail-user=mbui@mail.uni-mannheim.de
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

now=$(date +"%T")
echo "Program starts:  $now"

# Activate conda env
# srun $1
# Run script
srun python /home/ma/ma_ma/$SLURM_JOB_USER/clkd/run.py students/individual/loss=tinybertloss

end=$(date +"%T")
echo "Completed: $end"