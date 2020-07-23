# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:30:27 2019

@author: Greg0
"""

import numpy as np
from scipy._lib._util import check_random_state, MapWrapper  
import retro
import keyboard
import csv

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')#, record=True)

#qprint(env.)
env.reset()
action = [0,0,0,0,0,0,0,1,0,0,0,0]
dilatacion = 12

reader1 = csv.reader(open('Level1-260m-12d-20p-2P\Moves.csv'), delimiter=",")
x = list(reader1)
for i in range(1, int(len(x)/2+1) ):
    del x[i]

movement = np.array(x)#, dtype=bool)#.astype("bool")
movement = movement == "True"

size = np.shape(movement)

for i in range(0, size[0]):
    env.reset()
    for offset in range(0, int(size[1]/3) ):
    
        #action = env.action_space.sample()
        action[6] = movement[i, offset*3 + 0];
        action[7] = movement[i, offset*3 + 1];
        action[8] = movement[i, offset*3 + 2];
        
        for d in range(0, dilatacion):
            ob, rew, done, info = env.step(action)
            env.render()
        if(info["lives"] <= 1 or done):
            break
        
        if(keyboard.is_pressed('q')):
            break
env.close()
