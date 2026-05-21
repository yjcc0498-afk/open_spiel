#!/bin/bash

#SBATCH --job-name=elin_quad
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
python evaluation_example.py --game_name=mean_field_lin_quad --game_size=10 --game_horizon=10 --meta_strategy_method=nash --oracle_type=BR --IL_iterations=200 --egta_iterations=9  --root_result_folder=mfg_linq_RD_dist_eval

