## How to run the game model learning experiment?
1. Sample data from a MFG. Run generate_data.py.
2. Train a regressor. Use the generated data to train a regressor by running tf_regression.py.
3. Evalute the learned regressor. Training/testing metrics have been printed out. 
Run evaluation_example.py to compute the NE with the learned model and then the corresponding regret.