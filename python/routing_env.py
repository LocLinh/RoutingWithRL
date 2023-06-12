from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
from gymnasium import Env
import numpy as np
import constant
import pygame

class Routing(Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, size=5):
        self.action_space = Discrete(4) # 0: left, 1: right, 2: top, 3: bottom
        self.state = np.loadtxt(constant.MAP_PATH, delimiter='\t')
        self.observation_space = Box(low=0, high=4, shape=self.state.shape, dtype=int)            
        start = np.where(self.state == 1)
        self.start_point = (start[0][0], start[1][0])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None

    def render(self):
        pass

    def step(self, action:int):
        reward = -1
        done = False
        truncate = False
        info = {}

        my_position = np.where(self.state == constant.START_POINT)
        my_position = (my_position[0][0], my_position[1][0])
        left_block, right_block, top_block, bottom_block = self._get_surrounding_blocks(my_position)
        match action:
            case 0: # left
                if left_block == None or left_block == constant.BIN:
                    return self.state, reward, done, truncate, info
                if left_block == constant.ROAD:
                    self.state[my_position[0], my_position[1]] = 0
                    self.state[my_position[0], my_position[1]-1] = 1
                    my_position = (my_position[0], my_position[1]-1)
            case 1: # right
                if right_block == None or right_block == constant.BIN:
                    return self.state, reward, done, truncate, info
                if right_block == constant.ROAD:
                    self.state[my_position[0], my_position[1]] = 0
                    self.state[my_position[0], my_position[1]+1] = 1
                    my_position = (my_position[0], my_position[1]+1)
            case 2: # top
                if top_block == None or top_block == constant.BIN:
                    return self.state, reward, done, truncate, info
                if top_block == constant.ROAD:
                    self.state[my_position[0], my_position[1]] = 0
                    self.state[my_position[0]-1, my_position[1]] = 1
                    my_position = (my_position[0]-1, my_position[1])
            case 3: # bottom
                if bottom_block == None or bottom_block == constant.BIN:
                    return self.state, reward, done, truncate, info
                if bottom_block == constant.ROAD:
                    self.state[my_position[0], my_position[1]] = 0
                    self.state[my_position[0]+1, my_position[1]] = 1
                    my_position = (my_position[0]+1, my_position[1])
            case _:
                self.state[my_position[0], my_position[1]] = 1
        
        
        # new position 
        left_block, right_block, top_block, bottom_block = self._get_surrounding_blocks(my_position)
        if left_block == constant.PICKUP_ITEM:
            self.state[my_position[0], my_position[1]-1] = constant.BIN
            reward += constant.REWARD_POINT
        if right_block == constant.PICKUP_ITEM:
            self.state[my_position[0], my_position[1]+1] = constant.BIN
            reward += constant.REWARD_POINT

        # check end game
        pickup_locations = np.where(self.state == constant.PICKUP_ITEM)
        if len(pickup_locations[0]) == 0:
            if my_position != self.start_point:
                if self.state[self.start_point[0], self.start_point[1]] != constant.PICKUP_ITEM:
                    self.state[self.start_point[0], self.start_point[1]] = constant.PICKUP_ITEM
            else:
                done = True

        return self.state, reward, done, truncate, info
    
    def reset(self, seed=None):
        self.state = np.loadtxt(constant.MAP_PATH, delimiter='\t')
        start = np.where(self.state == 1)
        self.start_point = (start[0][0], start[1][0]) 
        info = {}
        return self.state, info
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def _get_surrounding_blocks(self, my_position):
        
        left   = None if my_position[1] <= 0 else self.state[my_position[0], my_position[1]-1]
        right  = None if my_position[1] >= self.state.shape[1]-1 else self.state[my_position[0], my_position[1]+1]
        top    = None if my_position[0] <= 0 else self.state[my_position[0]-1, my_position[1]]
        bottom = None if my_position[0] >= self.state.shape[0]-1 else self.state[my_position[0]+1, my_position[1]]
        return left, right, top, bottom