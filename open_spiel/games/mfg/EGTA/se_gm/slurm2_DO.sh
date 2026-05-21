#!/bin/bash

#SBATCH --job-name=mfg_crowd
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
python egta_example.py --game_name=mfg_crowd_modelling --game_size=100 --game_horizon=30 --meta_strategy_method=RD --oracle_type=BR --IL_iterations=50 --egta_iterations=25 --w_distance=0.015 --planning_iters=10 --fine_tune_iters=40 --root_result_folder=mfg_crowd_modelling_1d_fixed_iter

