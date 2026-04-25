#!/bin/bash

#SBATCH --job-name=mfg_pred
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=14g
#SBATCH --time=05-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python egta_example.py --game_name=python_mfg_predator_prey --game_size=5 --game_horizon=10 --meta_strategy_method=nash --oracle_type=BR --IL_iterations=200 --egta_iterations=50  --root_result_folder=python_mfg_predator_prey_IL200

