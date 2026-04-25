#!/bin/bash

#SBATCH --job-name=mfg_crowd
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
python egta_example.py --game_name=mfg_crowd_modelling --meta_strategy_method=RD --oracle_type=BR --egta_iterations=25 --w_distance=0.015 --planning_iters=7 --fine_tune_iters=23 --game_size=100 --game_horizon=20 --IL_iterations=30 --root_result_folder=baseline_mfg_crowd_modelling_RD30_size100_step20