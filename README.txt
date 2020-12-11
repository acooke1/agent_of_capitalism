WELCOME TO AGENT OF CAPITALISM!

This is a game that we programmed and built several models for. The game is simple. Maps can be randomly generated or some are saved
in game_level.py. Each level is filled with coins and once all the coins have been picked up, the game is won. There is also an 
enemy that can be added, and touching this enemy will end the game early. 

We have four different models. A Q-learning model, a reinforce model, a reinforce with baseline, and a ppo model. In order to run
the q-learning model, run: python q-learning_model.py. In order to run any of the other 3, using: python agentCapt.py ____, where
the last argument is one of: "REINFORCE", "REINFORCE_BASELINE", or "PPO" respectively. This will train the model with the current
hyperparameters and print the results. Enjoy!