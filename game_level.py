import numpy as np
import tensorflow as tf


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
    def __init__(self, level):
        """
        :param level: the integer corresponding to which level map we will be using
        """
        self.level_map = levels[level][:][:] # These [:] indexing is used so that the level_map is a copy of the original, not a reference
        # Should we convert the list into a numpy array?
        self.num_coins_left = level_num_coins[level]
        self.player_pos = [1,1] # Index corresponding to the player's current location in the map


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

        goal_pos = self.player_pos[:] # Again, making a copy, not a reference
        if action==0:
            goal_pos[1] += -1 # move left
        elif action==1:
            goal_pos[0] += -1 # move up
        if action==2:
            goal_pos[1] += 1 # move right
        elif action==3:
            goal_pos[0] += 1 # move down
        # TODO Add code to this statement if we implement attacking!

        goal_pos_contents = self.level_map[goal_pos[0]][goal_pos[1]]
        if goal_pos_contents == 1: # moving into a wall
            reward = -1 # TODO should we penalize running into walls?
            goal_pos = self.player_pos[:] # The player can't move into a wall, so it stays stationary
        if goal_pos_contents == 2: # picking up a coin!
            self.num_coins_left -= 1
            reward = 1

        if self.num_coins_left == 0:
            reward = 10
            done = True

        # update player position
        self.level_map[self.player_pos[0]][self.player_pos[1]] = 0 # remove the player from the previous space in the map
        self.level_map[goal_pos[0]][goal_pos[1]] = 3 # add the player to the new space in the map
        self.player_pos = goal_pos[:]

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
        num_to_char = [" ", "█", "❂", "☃"]
        side_length = len(self.level_map)
        for y in range(side_length):
            print_string = ""
            for x in range(side_length):
                print_string += num_to_char[self.level_map[y][x]]
            print(print_string)


# = = = = TEST CODE = = = =

def main():
    level = GameLevel(0)
    level.print_map()
    level.step(2)
    level.print_map()
    level.step(2)
    level.print_map()
    level.step(2)
    level.print_map()
    level.step(0)
    level.print_map()
    level.step(3)
    level.print_map()


if __name__ == '__main__':
    main()