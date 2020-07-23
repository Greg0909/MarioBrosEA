import retro
import keyboard
import numpy as np



env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')#, record=True)

#qprint(env.)
env.reset()
action = [0,0,0,0,0,0,0,1,0,0,0,0]

done = False
while not done:
    env.render()
    
    #action = env.action_space.sample()

    
    ob, rew, done, info = env.step(action)
    print("Action ", action, "Reward ", rew, "Info ", info)
    
    if(info["time"] < 396 and info["time"] > 393):
        action = [0,0,0,0,0,0,0,1,0,0,0,0]
    if(info["time"] < 393 and info["time"] > 380):
        action = [0,0,0,0,0,0,0,1,1,0,0,0]
    
    #if(info["time"] < 385):
    #   env.render()
    if(keyboard.is_pressed('q')):
        break

env.close()
print("done")