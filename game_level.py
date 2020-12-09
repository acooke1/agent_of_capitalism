import numpy as np
import tensorflow as tf
import copy
import random


# NOTE FOR DEVELOPMENT:
# 0 is an empty space
# -0.1 is a wall
# 1.0 is a coin
# 0.1 is the player
# -1.0 is the enemy
levels = [
    [
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
    def __init__(self, level, has_enemy=False, use_submap=False, use_random_maps=False, side_length=16, wall_prop=0.3, num_coins=12, starting_pos=[1,1], use_random_starts=True):
        """
        :param level: the integer corresponding to which level map we will be using
        :param has_enemy: boolean corresponding to whether or not an enemy will be used; defaults to False
        :param use_submap: boolean corresponding to whether or not the level should return the whole level map
            or just a self.submap_dims square submap centered on the player when step() is called
        :param use_random_maps: boolean corresponding to whether the level should generate a new random map every time it is reset
            If True, the procedural generation uses the following parameters below
        :params side_length, wall_prop, num_coins, starting_pos, random_starting_pos: all to be passed to procedurally_generate()
        """
        # What walls/coins/etc. look like, to the model
        # DO NOT CHANGE, or else the model just won't work
        self.empty_level_val = 0.0 # DO NOT CHANGE
        self.wall_level_val = -0.1 # DO NOT CHANGE
        self.coin_level_val = 1.0 # DO NOT CHANGE
        self.player_level_val = 0.1 # DO NOT CHANGE
        self.enemy_level_val = -1.0 # DO NOT CHANGE

        # Store the game initialization parameters
        self.level_num = level
        self.use_submap = use_submap
        self.use_random_maps = use_random_maps
        self.use_random_starts = use_random_starts
        self.level_map = [] # levels[self.level_num].copy()
        self.player_pos = [] # of the format [y, x]
        self.has_enemy = has_enemy
        self.enemy_alive = False

        # Store level generation parameters (ONLY IF USING RANDOMLY GENERATED MAPS)
        self.side_length = side_length
        self.wall_prop = wall_prop
        self.num_coins = num_coins
        self.starting_pos = starting_pos

        # General usage constants
        self.num_direcs = 4
        self.coord_adds = [[0,-1], [-1,0], [0,1], [1,0]]
        self.level_area = self.side_length ** 2 # TEMPORARY, GETS RESET BELOW
        self.step_num = 0

        self.reset_level()

        # More general usage constants
        self.enemy_running_into_wall_factor = -0.8 # Lower values means less likely it will run into a wall (value must be >-1.0)
        self.enemy_pursuing_player_factor = 5.0 # Higher values means greater weight on pursuing the player
        self.submap_dims = 7
        self.level_area = len(self.level_map) ** 2
        self.max_steps = 2*self.level_area

        # Reward values
        # TODO tweak these values
        self.empty_space_reward = 0 # NOTE: CURRENTLY NOT IN USE--SEE LINE 162 FOR HOW REWARD FOR EMPTY SPACES IS CALCULATED
        self.hit_wall_reward = -.1
        self.get_coin_reward = 0.5
        self.get_all_coins_reward = 0
        self.slay_enemy_reward = 0.5
        self.get_hit_by_enemy_reward = -1.0

        self.reset_level()


    def reset_level(self):
        # Reinitialize the map
        
        # Reset step num and player position to starting position
        self.step_num = 0
        self.player_pos = copy.deepcopy(self.starting_pos)

        temp_enemy_pos = []
        if not self.use_random_maps:
            self.level_map = copy.deepcopy(levels[self.level_num])
            self.num_coins_left = copy.deepcopy(level_num_coins[self.level_num])
        else:
            self.level_map, num_generations, self.starting_pos, temp_enemy_pos = self.procedurally_generate(self.side_length, self.wall_prop, self.num_coins, self.starting_pos, self.use_random_starts)
            self.num_coins_left = copy.deepcopy(self.num_coins)
        
        # Add the player to the map
        self.level_map[self.player_pos[0]][self.player_pos[1]] = self.player_level_val

        # Reset any necessary enemy variables
        if self.has_enemy:
            if self.use_random_maps and self.use_random_starts:
                self.enemy_pos = temp_enemy_pos
            else:
                self.enemy_pos = [len(self.level_map)-2, len(self.level_map)-2]

            self.enemy_alive = True
            self.enemy_over_coin = (self.level_map[self.enemy_pos[0]][self.enemy_pos[1]]==self.coin_level_val)
            self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = self.enemy_level_val

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
        :returns: a list containing three things:
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
                old_pos_reward = self.get_coin_reward / self.num_steps_to_item(self.coin_level_val, self.player_pos)
                new_pos_reward = self.get_coin_reward / self.num_steps_to_item(self.coin_level_val, goal_pos)
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
                print("END CONDITION: hit by enemy")

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
            elif goal_pos_contents==self.wall_level_val: # attacking a wall
                reward = self.hit_wall_reward
            else:
                steps_to_enemy = self.num_steps_to_item(self.enemy_level_val, goal_pos) # distance from the attacked square to the enemy
                reward = self.slay_enemy_reward / (steps_to_enemy + 1)
                # Maximum reward is one half the value of hitting the enemy, and that's if it's right next to the player
        
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

            """
            print("Enemy direction probs for this step:")
            print(direction_probs)
            print("Action taken: " + str(enemy_action))
            """

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
                print("END CONDITION: hit by enemy")
            
            # Update enemy position
            # Remove the enemy from the previous space in the map
            if self.enemy_over_coin:
                self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = self.coin_level_val
            else:
                self.level_map[self.enemy_pos[0]][self.enemy_pos[1]] = self.empty_level_val
            self.enemy_over_coin = enemy_will_be_over_coin

            self.level_map[enemy_goal_pos[0]][enemy_goal_pos[1]] = self.enemy_level_val # add the enemy to the new space in the map
            self.enemy_pos = enemy_goal_pos[:]

        # Non-enemy-related episode ending conditions
        if self.num_coins_left == 0:
            reward = self.get_all_coins_reward
            done = True
            print("END CONDITION: got all coins!")
        if self.step_num >= self.max_steps:
            done = True
            print("END CONDITION: ran out of time")

        return_map = self.level_map
        if self.use_submap:
            return_map = self.create_submap()
            # Test that it's rendering correctly
            """print("SUBMAP")
            self.print_map(return_map)"""
        flattened_map = np.array(return_map).flatten()
        #print(flattened.shape)

        # If episode is finished, reset level parameters for the start of the next episode
        if done:
            self.reset_level()

        return [flattened_map, reward, done]


    def reset(self):
        return_map = self.level_map
        if self.use_submap:
            return_map = self.create_submap()
        
        return np.array(return_map).flatten()


    def create_submap(self):
        """
        Returns a square two-dimensional array with shape self.submap_dims by self.submap_dims,
        which is a subsection of self.level_map, specifically centered on the player's current location.
        Any spaces beyond the edges of the board are filled in by walls.
        """
        return_map = [[0.0 for x in range(self.submap_dims)] for y in range(self.submap_dims)]
        half_dims = self.submap_dims//2
        for y in range(self.submap_dims):
            y_coord = min(max(self.player_pos[0]-half_dims+y, 0), len(self.level_map)-1)
            for x in range(self.submap_dims):
                x_coord = min(max(self.player_pos[1]-half_dims+x, 0), len(self.level_map)-1)
                return_map[y][x] = self.level_map[y_coord][x_coord]

        return return_map


    def num_steps_to_item(self, search_item, input_pos, map_to_search=[]):
        """
        Finds the number of spaces from the player to the nearest specified item.
        Essentially a breadth-first-search.
        :param search_item: a float corresponding to the item that is being sought out
        :param input_pos: a length-2 list of integers, indicating the position to start from
        :param map_to_search: a possible non-standard map to search
        :returns: an integer number of steps to the closest item
        """
        search_map = map_to_search
        if len(map_to_search)==0:
            search_map = self.level_map

        level_width = len(search_map)
        greatest_dist = self.level_area # this theoretically should be the greatest possible distance to any given item
        distances = [[greatest_dist for x in range(level_width)] for y in range(level_width)] 
        open_spaces = []
        open_spaces.append(input_pos)
        distances[input_pos[0]][input_pos[1]] = 0
        min_dist = self.level_area + 1

        while len(open_spaces)>0:
            """for i in range(8):
                print(distances[i])
            """

            cur_pos = open_spaces.pop(0)
            # Loop through the adjacent spaces
            for direc in range(self.num_direcs):
                new_pos = [cur_pos[0]+self.coord_adds[direc][0],cur_pos[1]+self.coord_adds[direc][1]]
                pos_contents = search_map[new_pos[0]][new_pos[1]]
                pos_visited = (distances[new_pos[0]][new_pos[1]] < greatest_dist)

                # If it's the search_item, then it must be the closest one: we're done
                if pos_contents == search_item:
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

        if search_item==self.coin_level_val:
            print("ERROR IN num_steps_to_item function: the next coin is not accessible.")

        return -1


    def procedurally_generate(self, side_length, wall_prop, num_coins, starting_pos=[1,1], random_starting_pos=True):
        """
        Method creates a two-dimensional list of shape [side_length, side_length] that contains walls and num_coins coins
        :param side_length: An integer, usually between 8 and 16
        :param wall_prop: A float from 0.0 to 1.0 corresponding to the fraction of spaces within the level that should be walls
            Generally, 0.27 is a safe setting
        :param num_coins: An integer, usually around 8 to 10
        :param starting_pos: A length-two list of integers, indicating the position in the map at which the player should start
        :param random_starting_pos: boolean corresponding to whether or not to choose random player and enemy starting positions
            Defaults to True because the enemy may not work if it does not get a random (accessible) starting position
        :returns: a list containing four things:
            a two-dimensional list as described above
            an integer for the number of maps the function needed to generate before getting a usable map
            a length-two list of integers for the player starting position
            a length-two list of integers for the enemy starting position
        """
        accessible = False
        num_generations = 0
        gen_map = []

        while not accessible:
            num_generations+=1
            # Create the basic empty level with walls around the outside
            gen_map = []
            gen_map.append([])
            for x in range(side_length):
                gen_map[0].append(self.wall_level_val)
            for y in range(1, side_length-1):
                gen_map.append([self.wall_level_val])
                for x in range(side_length-2):
                    gen_map[y].append(self.empty_level_val)
                gen_map[y].append(self.wall_level_val)
            gen_map.append([])
            for x in range(side_length):
                gen_map[side_length-1].append(self.wall_level_val)

            player_start_pos = starting_pos
            if random_starting_pos:
                player_start_pos = [random.randint(1,side_length-2), random.randint(1,side_length-2)]

            # Add player placeholder
            gen_map[player_start_pos[0]][player_start_pos[1]] = self.player_level_val

            # Add some walls
            # (Note: counting the walls in all the example maps, while excluding the external walls, 
            # and then dividing by the squared internal area 6x6 for an 8x8 map, 14x14 for a 16x16 map,
            # we get a proportion of around 0.27 oor 0.28)
            num_walls = int(wall_prop*((side_length-2)**2))
            for i in range(num_walls):
                # Find a random open location
                wall_x = random.randint(1,side_length-2)
                wall_y = random.randint(1,side_length-2)
                while not gen_map[wall_y][wall_x]==self.empty_level_val:
                    wall_x = random.randint(1,side_length-2)
                    wall_y = random.randint(1,side_length-2)

                # Add the wall to the map
                gen_map[wall_y][wall_x] = self.wall_level_val

            # Add some coins
            for i in range(num_coins):
                # Find a random open location
                coin_x = random.randint(1,side_length-2)
                coin_y = random.randint(1,side_length-2)
                while not gen_map[coin_y][coin_x]==self.empty_level_val:
                    coin_x = random.randint(1,side_length-2)
                    coin_y = random.randint(1,side_length-2)

                # Add the coin to the map
                gen_map[coin_y][coin_x] = self.coin_level_val

            # print("MAP GENERATION " + str(num_generations))
            # self.print_map(gen_map)

            # Now, check if all the coins are accessible
            unvisited = [[True for x in range(side_length)] for y in range(side_length)] 
            open_spaces = []
            open_spaces.append(player_start_pos)
            unvisited[player_start_pos[0]][player_start_pos[1]] = False
            num_coins_found = 0

            # Loop through all available paths
            while len(open_spaces)>0:
                cur_pos = open_spaces.pop(0)
                # Loop through the adjacent spaces
                for direc in range(self.num_direcs):
                    new_pos = [cur_pos[0]+self.coord_adds[direc][0],cur_pos[1]+self.coord_adds[direc][1]]

                    if unvisited[new_pos[0]][new_pos[1]]:
                        pos_contents = gen_map[new_pos[0]][new_pos[1]]

                        if pos_contents == self.coin_level_val:
                            num_coins_found += 1
                        
                        if not pos_contents == self.wall_level_val:
                            open_spaces.append(new_pos)
                            unvisited[new_pos[0]][new_pos[1]] = False

            # print("Looping through the map found " + str(num_coins_found) + " coins")
            # Test to see if all the coins are accessible
            accessible = (num_coins_found==num_coins)

        # Once an accessible map has been generated, we can find an accessible enemy_start_pos
        enemy_start_pos = [random.randint(1,side_length-2), random.randint(1,side_length-2)]
        bad_enemy_pos = True
        while bad_enemy_pos:
            # Check if the position puts the enemy in an empty square
            pos_contents = gen_map[enemy_start_pos[0]][enemy_start_pos[1]]
            if not (pos_contents==self.wall_level_val or pos_contents==self.enemy_level_val):
                # Check if the enemy can reach the player from this position
                steps_to_player = self.num_steps_to_item(self.player_level_val, enemy_start_pos, map_to_search=gen_map)
                if steps_to_player>2:
                    bad_enemy_pos = False

            if bad_enemy_pos:
                enemy_start_pos = [random.randint(1,side_length-2), random.randint(1,side_length-2)]

        # Add enemy placeholder
        enemy_over_coin = (gen_map[enemy_start_pos[0]][enemy_start_pos[1]]==self.coin_level_val)
        gen_map[enemy_start_pos[0]][enemy_start_pos[1]] = self.enemy_level_val

        """
        # UNCOMMENT THIS TO GET MAP GENERATION STATS
        self.print_map(gen_map)
        print(str(num_generations) + " MAPS GENERATED TO GET AN ACCESSIBLE MAP")
        if enemy_over_coin:
            print("ENEMY IS OVER COIN")
        """

        # Remove player/enemy placeholder
        gen_map[player_start_pos[0]][player_start_pos[1]] = self.empty_level_val
        if enemy_over_coin:
            gen_map[enemy_start_pos[0]][enemy_start_pos[1]] = self.coin_level_val
        else:
            gen_map[enemy_start_pos[0]][enemy_start_pos[1]] = self.empty_level_val

        return [gen_map, num_generations, player_start_pos, enemy_start_pos]

    def print_map(self, map_to_print=[]):
        """
        Method should print the currently-stored self.level_map to console
        :param map_to_print: A nxn array of float values corresponding to a map that should be printed
        :returns: None
        """
        if len(map_to_print)==0:
            map_to_print = self.level_map

        print("Step number " + str(self.step_num))
        num_to_char = [" ", "█", "❂", "♀", "☿"] # Want to go back to the snowman? He's here → ☃
        side_length = len(map_to_print)
        for y in range(side_length):
            print_string = ""
            for x in range(side_length):
                space_contents = map_to_print[y][x]
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

def main():

    # PARAMETERS FOR THIS TRAINING RUN
    game_level = 3
    use_submap = True
    use_enemy = True

    # PARAMETERS FOR RANDOM MAP GENERATION
    use_random_maps = False
    side_length = 16 # Generally, values between 8 and 16 are good
    wall_prop = 0.3 # This is the fraction of empty spaces that become walls. Generally, values between 0.25 and 0.35 are good
    num_coins = 12
    starting_pos = [1,1] # Setting this to [1,1] is standard (top-left corner), but if you wanted, you could set it to [4,5], or other starting positions
    use_random_starts = True

    level = GameLevel(game_level, use_enemy, use_submap, use_random_maps, side_length, wall_prop, num_coins, starting_pos, use_random_starts)
    # level.print_map()

    # TEST PARAMETERS FOR MASS GENERATION
    """
    side_length = 16
    num_coins = 12
    wall_prop = 0.3
    num_generations = []
    for i in range(100):
        genned_map, num_gens, player_pos, enemy_pos = level.procedurally_generate(side_length=side_length, wall_prop=wall_prop, num_coins=num_coins, use_random_starts=True)
        num_generations.append(num_gens)
    print("Side length " + str(side_length) + "; wall prop " + str(wall_prop) + "; num coins " + str(num_coins) + ": " + str(sum(num_generations)/len(num_generations)) + " maps need to be generated on average")
    """

    # COMMENT OUT ABOVE CODE TO GET THE TEST MOVEMENTS IN THE MAP
    
    test_steps = [2, 2, 2, 2, 2, 2, 2, 3, 3, 0, 3, 3, 3]
    # A good set of test_steps for level 0: [2, 2, 2, 0, 3, 3, 0, 0, 2, 3]

    # Print initial board state...
    if use_submap:
        level.print_map(map_to_print=level.create_submap())
    else:
        level.print_map()
    # Run the steps in the test_steps list!
    for i in range(len(test_steps)):
        print("Distance to coin: " + str(level.num_steps_to_item(level.coin_level_val, level.player_pos)))
        level.step(test_steps[i])
        if use_submap:
            level.print_map(map_to_print=level.create_submap())
        else:
            level.print_map()
    

if __name__ == '__main__':
    main()
