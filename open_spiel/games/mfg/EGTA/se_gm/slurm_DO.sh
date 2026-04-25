#!/bin/bash

#SBATCH --job-name=mfg_pred
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=14g
#SBATCH --time=07-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python egta_example.py --game_name=mean_field_lin_quad --game_size=10 --game_horizon=10 --meta_strategy_method=RD --oracle_type=BR --IL_iterations=200 --egta_iterations=9 --w_distance=0.015 --planning_iters= 150 --fine_tune_iters=50 --root_result_folder=mfg_linq

