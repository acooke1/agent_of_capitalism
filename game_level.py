import numpy as np
import tensorflow as tf
import copy


# NOTE FOR DEVELOPMENT: it's probably really bad practice to store this levels variable as a global variable
# 0 is an empty space
# -0.1 is a wall
# 1.0 is a coin
# 0.1 is the player
# -1.0 is the enemy (if implemented)
levels = [
    [ # this first level is probably a bad level to try to use an enemy on: lots of choke points to get stuck in, hard to get around the enemy
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1],
        [-.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -.1],
        [-.1, -.1, -.1, 0.0, -.1, 0.0, 0.0, -.1],
        [-.1, 1.0, 0.0, 0.0, -.1, 1.0, 1.0, -.1],
        [-.1, -.1, 0.0, 0.0, -.1, -.1, -.1, -.1],
        [-.1, -.1, -.1, 0.0, 0.0, 0.0, 1.0, -.1],
        [-.1, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -.1],
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1]
    ],
    [
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1],
        [-.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, 0.0, -.1, 0.0, 0.0, -.1, 0.0, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, 1.0, 0.0, 0.0, -.1, 0.0, 0.0, -.1, 0.0, 0.0, 1.0, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, 0.0, 0.0, 0.0, -.1, 0.0, 0.0, -.1, 0.0, 0.0, 0.0, -.1, 0.0, -.1],
        [-.1, 0.0, 0.0, 0.0, 0.0, 0.0, -.1, 0.0, 0.0, -.1, 0.0, 0.0, 0.0, 0.0, 0.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, -.1, -.1, 0.0, 0.0, -.1, -.1, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -.1],
        [-.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, -.1, -.1, 0.0, 0.0, -.1, -.1, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 0.0, 0.0, 0.0, 0.0, 0.0, -.1, 0.0, 0.0, -.1, 0.0, 0.0, 0.0, 0.0, 0.0, -.1],
        [-.1, 0.0, -.1, 0.0, 0.0, 0.0, -.1, 0.0, 0.0, -.1, 0.0, 0.0, 0.0, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, 1.0, 0.0, 0.0, -.1, 0.0, 0.0, -.1, 0.0, 0.0, 1.0, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, 0.0, -.1, 0.0, 0.0, -.1, 0.0, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -.1],
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1]
    ],
    [
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1],
        [-.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 1.0, 0.0, 0.0, -.1, 1.0, 0.0, -.1],
        [-.1, 0.0, -.1, 1.0, -.1, 0.0, -.1, -.1],
        [-.1, 0.0, -.1, 0.0, -.1, 0.0, 1.0, -.1],
        [-.1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -.1],
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1]
    ],
    [
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1],
        [-.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -.1, 1.0, 1.0, 0.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, 0.0, -.1, -.1, 0.0, -.1, 0.0, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, 1.0, 0.0, 0.0, -.1, 1.0, 0.0, -.1, 0.0, 0.0, 1.0, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, -.1, -.1, 0.0, -.1, -.1, -.1, -.1, 0.0, -.1, 0.0, -.1],
        [-.1, 0.0, 0.0, 0.0, 0.0, 0.0, -.1, 0.0, -.1, 0.0, 0.0, 0.0, 0.0, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, 0.0, -.1, 0.0, -.1, 0.0, -.1, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, 1.0, -.1, 0.0, -.1, 0.0, -.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -.1],
        [-.1, 0.0, 0.0, 0.0, -.1, 0.0, 0.0, 0.0, -.1, -.1, -.1, -.1, -.1, -.1, 0.0, -.1],
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, 0.0, -.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -.1],
        [-.1, 0.0, 0.0, 0.0, -.1, 0.0, 0.0, 0.0, -.1, 0.0, -.1, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 0.0, -.1, 0.0, -.1, 0.0, -.1, 1.0, 0.0, 0.0, -.1, 1.0, 1.0, 0.0, 0.0, -.1],
        [-.1, 0.0, -.1, 1.0, 0.0, 0.0, -.1, -.1, -.1, -.1, -.1, -.1, 1.0, 0.0, 0.0, -.1],
        [-.1, 0.0, -.1, -.1, -.1, -.1, -.1, 0.0, 0.0, 0.0, 0.0, -.1, -.1, -.1, 0.0, -.1],
        [-.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -.1, -.1, 0.0, 0.0, 0.0, 0.0, 0.0, -.1],
        [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1]
    ]
]
level_num_coins = [8, 11, 7, 13]


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
        self.num_direcs = 4
        self.coord_adds = [[0,-1], [-1,0], [0,1], [1,0]]
        self.enemy_running_into_wall_factor = 0.9 # Higher values means less likely it will run into a wall
        self.enemy_pursuing_player_factor = 0.5 # Higher values means greater weight on pursuing the player

        # What walls/coins/etc. look like, to the model
        # DO NOT CHANGE, or else the model just won't work
        self.empty_level_val = 0.0 # DO NOT CHANGE
        self.wall_level_val = -0.1 # DO NOT CHANGE
        self.coin_level_val = 1.0 # DO NOT CHANGE
        self.player_level_val = 0.1 # DO NOT CHANGE
        self.enemy_level_val = -1.0 # DO NOT CHANGE

        # Reward values
        # TODO tweak these values
        self.empty_space_reward = 0
        self.hit_wall_reward = 0
        self.get_coin_reward = .5
        self.get_all_coins_reward = 10
        self.slay_enemy_reward = 0.5
        self.get_hit_by_enemy_reward = -1.0

        self.reset_level()
        self.state_size = len(self.level_map) ** 2
        self.max_steps = 2*self.state_size

    def reset_level(self):
        self.level_map = copy.deepcopy(levels[self.level_num])
        self.num_coins_left = copy.deepcopy(level_num_coins[self.level_num])
        self.player_pos = [1,1] # Index corresponding to the player's current location in the map
        # TODO TESTING STARTING IN THE MIDDLE
        # self.player_pos = [len(self.level_map)//2, len(self.level_map)//2] # Uncomment here if you want to go back to starting in the middle
        
        self.level_map[self.player_pos[0]][self.player_pos[1]] = self.player_level_val
        self.step_num = 0
        if self.has_enemy:
            self.enemy_pos = [len(self.level_map)-2, len(self.level_map)-2]
            self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = self.enemy_level_val
            self.enemy_alive = True
            self.enemy_over_coin = True

    """
    def coinsLeft(self, coin_value):
        total = 0
        for i in range(len(self.level_map)):
            total+=self.level_map[i].count(coin_value)
        return total
    """

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

        # Determines where the model's next move will bring it
        action_direc = action % self.num_direcs # This is important, since actions may be 4, 5, 6, 7 for attacking
        goal_pos = [self.player_pos[0]+self.coord_adds[action_direc][0], self.player_pos[1]+self.coord_adds[action_direc][1]]
        goal_pos_contents = self.level_map[goal_pos[0]][goal_pos[1]]

        if action<self.num_direcs:
            # Move validation: also determines reward in this step
            if goal_pos_contents == self.empty_level_val: # moving into an empty space
                # TODO tried swapping for the numsteps-based reward!
                old_pos_reward = self.get_coin_reward / self.num_steps_to_coin(self.player_pos)
                new_pos_reward = self.get_coin_reward / self.num_steps_to_coin(goal_pos)
                # With this calculation, the highest new_pos_reward for an empty space is the value of getting a coin
                # And if it was moving closer to the coin, then old_pos_reward should be half the value of getting the coin
                reward = new_pos_reward-old_pos_reward
                # ideally, new_pos_reward should be greater than old_pos_reward:
                # if not, the reward should be negative, for moving away from the nearest coin
                
                # TODO old reward value below
                # reward = self.empty_space_reward
            if goal_pos_contents == self.wall_level_val: # moving into a wall
                reward = self.hit_wall_reward
                goal_pos = self.player_pos # The player can't move into a wall, so it stays stationary
            if goal_pos_contents == self.coin_level_val: # picking up a coin!
                self.num_coins_left -= 1
                reward = self.get_coin_reward
            if goal_pos_contents == self.enemy_level_val: # running into the enemy
                reward = self.get_hit_by_enemy_reward
                done = True

            # Update player position
            self.level_map[self.player_pos[0]][self.player_pos[1]] = 0 # remove the player from the previous space in the map
            self.level_map[goal_pos[0]][goal_pos[1]] = self.player_level_val # add the player to the new space in the map
            self.player_pos = goal_pos[:]
        else:
            # Attack validation
            if goal_pos_contents==self.enemy_level_val: # Hit the enemy!
                reward = self.slay_enemy_reward
                self.enemy_alive = False
                # Remove the enemy from its previous space in the map
                if self.enemy_over_coin:
                    self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = self.coin_level_val
                    self.enemy_over_coin = False
                else:
                    self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = self.empty_level_val
            # If we want to add a negative reward for not hitting the enemy, here's the spot to do it, in an else statement
        
        # Enemy movement code
        if self.enemy_alive:
            # Generate randomness in direction choice
            direction_probs = [1.0] * self.num_direcs # Creates a list [1.0, 1.0, 1.0, 1.0]. Change the 1.0 if we want a different probability initialization
            y_difference = self.player_pos[0]-self.enemy_pos[0]
            x_difference = self.player_pos[1]-self.enemy_pos[1]
            direc_comparisons = [x_difference<0, y_difference<0, x_difference>0, y_difference>0]

            for direc in range(self.num_direcs):
                pos_contents = self.level_map[self.enemy_pos[0]+self.coord_adds[direc][0]][self.enemy_pos[1]+self.coord_adds[direc][1]]
                if pos_contents == self.wall_level_val: # That's a wall
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
            enemy_will_be_over_coin = (enemy_goal_pos_contents==self.coin_level_val)

            if enemy_goal_pos_contents == self.wall_level_val: # moving into a wall
                enemy_goal_pos = self.enemy_pos # The player can't move into a wall, so it stays stationary
                enemy_will_be_over_coin = self.enemy_over_coin
            if enemy_goal_pos_contents == self.player_level_val: # hitting the player
                reward = self.get_hit_by_enemy_reward
                done = True
            
            # Update enemy position
            # Remove the enemy from the previous space in the map
            if self.enemy_over_coin:
                self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = self.coin_level_val
            else:
                self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = self.empty_level_val
            self.enemy_over_coin = enemy_will_be_over_coin

            self.level_map[enemy_goal_pos[0]][enemy_goal_pos[1]] = self.enemy_level_val # add the enemy to the new space in the map
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

        flattened = np.array(self.level_map).flatten()
        #print(flattened.shape)
        return [flattened, reward, done]

    def reset(self):
        return np.array(self.level_map).flatten()

    def num_steps_to_coin(self, input_pos):
        """
        Finds the number of spaces from the player to the nearest coin.
        Essentially a breadth-first-search.
        :input_pos: a length-2 list of integers, indicating the position to start from
        :returns: an integer number of steps to the closest coin
        """
        level_width = len(self.level_map)
        greatest_dist = level_width*2 # ((level_width - 2)**2) + 1
        # TODO two times the side width theoretically should be the greatest possible distance to any given coin
        distances = [[greatest_dist for x in range(level_width)] for y in range(level_width)] 
        open_spaces = []
        open_spaces.append(input_pos)
        distances[input_pos[0]][input_pos[1]] = 0
        min_dist = self.state_size + 1

        while len(open_spaces)>0:
            """for i in range(8):
                print(distances[i])
            """

            cur_pos = open_spaces.pop(0)
            # Loop through the adjacent spaces
            for direc in range(self.num_direcs):
                new_pos = [cur_pos[0]+self.coord_adds[direc][0],cur_pos[1]+self.coord_adds[direc][1]]
                pos_contents = self.level_map[new_pos[0]][new_pos[1]]
                pos_visited = (distances[new_pos[0]][new_pos[1]] < greatest_dist)

                # If it's a coin, then it must be the closest coin: we're done
                if pos_contents == self.coin_level_val:
                    min_dist = distances[cur_pos[0]][cur_pos[1]]+1
                    return min_dist # TODO THIS MAY BE BAD PRACTICE
                elif not pos_contents == self.wall_level_val:
                    # If it's not a coin and not a wall, then we want to check the next available and unvisited spaces
                    if not pos_visited:
                        open_spaces.append(new_pos)
                    # Compare path length to this space through cur_pos
                    new_dist = distances[cur_pos[0]][cur_pos[1]]+1
                    if new_dist < distances[new_pos[0]][new_pos[1]]:
                        # Update the new shortest distance to that path
                        distances[new_pos[0]][new_pos[1]] = new_dist

        print("Error in the dist_to_coin function: code should never have reached this point.")
        return None


    def print_map(self):
        """
        Method should print the currently-stored self.level_map to console
        :returns: None
        """
        print("Step number " + str(self.step_num))
        num_to_char = [" ", "█", "❂", "♀", "☿"] # Want to go back to the snowman? He's here → ☃
        side_length = len(self.level_map)
        for y in range(side_length):
            print_string = ""
            for x in range(side_length):
                space_contents = self.level_map[y][x]
                if space_contents == self.empty_level_val:
                    char_to_add = num_to_char[0]
                elif space_contents == self.wall_level_val:
                    char_to_add = num_to_char[1]
                elif space_contents == self.coin_level_val:
                    char_to_add = num_to_char[2]
                elif space_contents == self.player_level_val:
                    char_to_add = num_to_char[3]
                elif space_contents == self.enemy_level_val:
                    char_to_add = num_to_char[4]
                print_string += char_to_add
            print(print_string)


# = = = = TEST CODE = = = =
"""
def main():
    level = GameLevel(0)
    level.print_map()
    print("Distance to coin: " + str(level.num_steps_to_coin(level.player_pos)))
    level.step(2)
    level.print_map()
    print("Distance to coin: " + str(level.num_steps_to_coin(level.player_pos)))
    level.step(2)
    level.print_map()
    print("Distance to coin: " + str(level.num_steps_to_coin(level.player_pos)))
    level.step(2)
    level.print_map()
    print("Distance to coin: " + str(level.num_steps_to_coin(level.player_pos)))
    level.step(0)
    level.print_map()
    print("Distance to coin: " + str(level.num_steps_to_coin(level.player_pos)))
    level.step(3)
    level.print_map()
    print("Distance to coin: " + str(level.num_steps_to_coin(level.player_pos)))


if __name__ == '__main__':
    main()
"""
