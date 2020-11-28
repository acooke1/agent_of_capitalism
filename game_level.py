import numpy as np
import tensorflow as tf
import copy


# NOTE FOR DEVELOPMENT: it's probably really bad practice to store this levels variable as a global variable
# 0 is an empty space
# 1 is a wall
# 2 is a coin
# 3 is the player
# 4 is the enemy (if implemented)
levels = [
    [ # this first level is probably a bad level to try to use an enemy on: lots of choke points to get stuck in, hard to get around the enemy
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 3, 0, 0, 2, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 0, 1],
        [1, 2, 0, 0, 1, 2, 2, 1],
        [1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 2, 1],
        [1, 2, 0, 0, 0, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ],
    [ # TODO if we implement the enemy, we can swap the bottom-right coin for the enemy
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 2, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 2, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
]
level_num_coins = [8, 11] # TODO tweak these values if we add an enemy and remove a coin


class GameLevel():
    def __init__(self, level, has_enemy=False):
        """
        :param level: the integer corresponding to which level map we will be using
        :param has_enemy: boolean corresponding to whether or not an enemy will be used; defaults to False
        """
        self.level_num = level
        self.level_map = [] # levels[self.level_num].copy()
        self.player_pos = []
        self.has_enemy = has_enemy
        self.enemy_alive = False

        # General usage constants
        self.coord_adds = [[0,-1], [-1,0], [0,1], [1,0]]
        self.enemy_running_into_wall_factor = 0.9 # Higher values means less likely it will run into a wall
        self.enemy_pursuing_player_factor = 0.5 # Higher values means greater weight on pursuing the player

        # Reward values
        # TODO tweak these values
        self.empty_space_reward = 0.01
        self.hit_wall_reward = -0.1
        self.get_coin_reward = 0.5
        self.get_all_coins_reward = 1.0
        self.slay_enemy_reward = 0.5
        self.get_hit_by_enemy_reward = -1.0

        self.reset_level()
        self.state_size = len(self.level_map) ** 2
        self.max_steps = self.state_size * 2

    def reset_level(self):
        #print("before")
        #self.print_map()
        self.level_map = copy.deepcopy(levels[self.level_num])
        #print("after")
        #self.print_map()
        self.num_coins_left = copy.deepcopy(level_num_coins[self.level_num])
        #print(self.coinsLeft(2))
        self.player_pos = [1,1] # Index corresponding to the player's current location in the map
        self.step_num = 0
        if self.has_enemy:
            self.enemy_pos = [len(self.level_map)-2, len(self.level_map)-2]
            self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = 4
            self.enemy_alive = True
            self.enemy_over_coin = True

    def coinsLeft(self, coin_value):
        total = 0
        for i in range(len(self.level_map)):
            total+=self.level_map[i].count(coin_value)
        return total

    def step(self, action):
        """
        :param action: an integer corresponding to an input action:
            0 - move left
            1 - move up
            2 - move right
            3 - move down
            4 - attack left (if we implement attacking)
            5 - attack up
            6 - attack right
            7 - attack down
        :returns: an list containing three things:
            an array corresponding to the current game state,
            an integer for the reward given for the action taken,
            a boolean for whether or not the game is done
        """
        reward = 0
        done = False
        self.step_num += 1 
        #print(self.level_map)

        # Determines where the model's next move will bring it
        """goal_pos = copy.deepcopy(self.player_pos)
        if action==0:
            goal_pos[1] += -1 # move left
        elif action==1:
            goal_pos[0] += -1 # move up
        if action==2:
            goal_pos[1] += 1 # move right
        elif action==3:
            goal_pos[0] += 1 # move down"""

        # Implemented a new fancy way of getting the goal_pos... hope this works
        action_direc = action % 4 # This is important, since actions may be 4, 5, 6, 7 for attacking
        goal_pos = [self.player_pos[0]+self.coord_adds[action_direc][0], self.player_pos[1]+self.coord_adds[action_direc][1]]
        goal_pos_contents = self.level_map[goal_pos[0]][goal_pos[1]]

        if action<4:
            # Move validation: also determines reward in this step
            if goal_pos_contents == 0: # moving into an empty space
                reward = self.empty_space_reward
            if goal_pos_contents == 1: # moving into a wall
                reward = self.hit_wall_reward
                goal_pos = self.player_pos # The player can't move into a wall, so it stays stationary
            if goal_pos_contents == 2: # picking up a coin!
                self.num_coins_left -= 1
                reward = self.get_coin_reward
            if goal_pos_contents == 4: # running into the enemy
                reward = self.get_hit_by_enemy_reward
                done = True

            # Update player position
            self.level_map[self.player_pos[0]][self.player_pos[1]] = 0 # remove the player from the previous space in the map
            self.level_map[goal_pos[0]][goal_pos[1]] = 3 # add the player to the new space in the map
            self.player_pos = goal_pos[:]
        else:
            # Attack validation
            if goal_pos_contents==4: # Hit the enemy!
                reward = self.slay_enemy_reward
                self.enemy_alive = False
                # Remove the enemy from its previous space in the map
                if self.enemy_over_coin:
                    self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = 2
                    self.enemy_over_coin = False
                else:
                    self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = 0
            # If we want to add a negative reward for not hitting the enemy, here's the spot to do it, in an else statement
        
        # Enemy movement code
        if self.enemy_alive:
            # Generate randomness in direction choice
            direction_probs = [1.0] * 4 # Creates a list [1.0, 1.0, 1.0, 1.0]. Change the 1.0 if we want a different probability initialization
            y_difference = self.player_pos[0]-self.enemy_pos[0]
            x_difference = self.player_pos[1]-self.enemy_pos[1]
            direc_comparisons = [x_difference<0, y_difference<0, x_difference>0, y_difference>0]

            for direc in range(4):
                pos_contents = self.level_map[self.enemy_pos[0]+self.coord_adds[direc][0]][self.player_pos[1]+self.coord_adds[direc][1]]
                if pos_contents == 1: # That's a wall
                    direction_probs[direc] += self.enemy_running_into_wall_factor
                elif direc_comparisons[direc]: # The player is in this direction
                    direction_probs[direc] += self.enemy_pursuing_player_factor

            # Normalize probabilities
            direction_probs = np.array(direction_probs)
            direction_probs = direction_probs / np.sum(direction_probs)

            # Enemy chooses a random action
            enemy_action = np.random.choice([0,1,2,3], p=direction_probs)

            # Enemy move validation
            enemy_goal_pos = [self.enemy_pos[0]+self.coord_adds[enemy_action][0], self.enemy_pos[1]+self.coord_adds[enemy_action][1]]
            enemy_goal_pos_contents = self.level_map[enemy_goal_pos[0]][enemy_goal_pos[1]]
            enemy_will_be_over_coin = (enemy_goal_pos_contents==2)

            if enemy_goal_pos_contents == 1: # moving into a wall
                enemy_goal_pos = self.enemy_pos # The player can't move into a wall, so it stays stationary
                enemy_will_be_over_coin = self.enemy_over_coin
            if enemy_goal_pos_contents == 3: # hitting the player
                reward = self.get_hit_by_enemy_reward
                done = True
            
            # Update enemy position
            # Remove the enemy from the previous space in the map
            if self.enemy_over_coin:
                self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = 2
            else:
                self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = 0
            self.enemy_over_coin = enemy_will_be_over_coin

            self.level_map[enemy_goal_pos[0]][enemy_goal_pos[1]] = 4 # add the enemy to the new space in the map
            self.enemy_pos = enemy_goal_pos[:]

        # Other episode ending conditions
        if self.num_coins_left == 0:
            reward = self.get_all_coins_reward
            done = True
        if self.step_num >= self.max_steps:
            done = True

        # If episode is finished, reset level parameters for the start of the next episode
        if done:
            self.reset_level()

        return [self.level_map, reward, done]


    def print_map(self):
        """
        Method should print the currently-stored self.level_map to console
        """
        """
        # This fancy method of using numpy array swaps ended up being way complicated to code, I'll just do it the messy for loop way
        text_map = np.array(self.level_map)
        text_map = text_map.astype("|S1")
        print(text_map)
        text_map[text_map=="0"] = " "
        text_map[text_map=="1"] = "X" # "█" isn't ascii unfortunately
        text_map[text_map=="2"] = "$" # "❂" probably isn't ascii 
        text_map[text_map=="3"] = "o" # "☃" isn't ascii
        # TODO choose a character and add a swap for the enemy!
        text_map = text_map.tolist()
        print(text_map)
        """
        print("Step number " + str(self.step_num))
        num_to_char = [" ", "█", "❂", "♀", "☿"] # Want to go back to the snowman? He's here → ☃
        side_length = len(self.level_map)
        for y in range(side_length):
            print_string = ""
            for x in range(side_length):
                print_string += num_to_char[self.level_map[y][x]]
            print(print_string)


# = = = = TEST CODE = = = =

def main():
    pass
    # level = GameLevel(0)
    # level.print_map()
    # level.step(2)
    # level.print_map()
    # level.step(2)
    # level.print_map()
    # level.step(2)
    # level.print_map()
    # level.step(0)
    # level.print_map()
    # level.step(3)
    # level.print_map()


if __name__ == '__main__':
    main()