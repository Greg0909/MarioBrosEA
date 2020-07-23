# -*- coding: utf-8 -*-
"""
Greg Espinosa A01196764
Mario Roque
"""

import numpy as np
from scipy._lib._util import check_random_state, MapWrapper  
import math
import retro
import keyboard
import csv

#
def genericEA(crossover = '2P', mutation = 'bitFlip', movimientos = 50, dilatacion = 1,
              generations = 1000, populationSize = 100, recombinationRate = 0.9, 
              mutationRate = 0.1, seed = None):
    
    with EvolutionaryAlgorithmSolver(crossover, mutation, movimientos, dilatacion,
                                     generations, populationSize, 
                                     recombinationRate, mutationRate, 
                                     seed) as solver:
        solution = solver.solve()
    
    return solution
    

class EvolutionaryAlgorithmSolver(object):
    
    def __init__(self, crossover = 'arithmetic', 
                 mutation = 'uniform', movimientos = 50, dilatacion = 1, generations = 10000, populationSize = 100, 
                 recombinationRate = 0.9, mutationRate = 0.1, seed = None):
        self.seed = check_random_state(seed)
        self.crossoverOp = crossover
        self.mutationOp = mutation
        self.generations = generations
        self.popsize = int(populationSize)
        self.recombinationrate = recombinationRate
        self.mutationrate = mutationRate
        self.movimientos = movimientos
        self.dilatacion = dilatacion
                
    def tournament(self, parents, fitness):
        rng = self.seed
        idx = np.argsort(rng.uniform(size = (self.popsize, self.popsize)))[:,0:4]
        idx_1 = fitness[idx[:,0]] < fitness[idx[:,1]]
        idx_2 = fitness[idx[:,2]] < fitness[idx[:,3]]
        idx_parent_1 = idx[:,1]
        idx_parent_1[idx_1] = idx[idx_1,0]
        idx_parent_2 = idx[:,3]
        idx_parent_2[idx_2] = idx[idx_2,2]
        return (parents[idx_parent_1,:], parents[idx_parent_2,:])
    
    def arithmetic(self, parents):
        rng = self.seed
        size = np.shape(parents[0])
        alpha = rng.uniform(size = size)
        mask = rng.choice([False, True], size = (size[0]), 
                          p = [self.recombinationrate, 1-self.recombinationrate])
        offspring = np.multiply(alpha, parents[0]) + np.multiply(1-alpha, 
                               parents[1])
        offspring[offspring > 1] = 1
        offspring[offspring < 0] = 0
        offspring[mask] = parents[0][mask]
        return offspring
    
    
    def uniform_crossover(self, parents):
        rnd = self.seed
        N = np.shape(parents[0])
        mask = rnd.choice(a=[False, True], size=(N), p=[0.5, 0.5])
        offspring = np.logical_or(np.logical_and(parents[0], mask), np.logical_and(parents[1], np.logical_not(mask)))
        return offspring

  
    def TwoPoint_crossover(self, parents):
        rnd  = self.seed
        N    = np.shape(parents[0])
        point1 = rnd.randint(0, N[1]-2, size=(N[0]))
        point2 = rnd.randint(point1+1, N[1], size=(N[0]))
        mask = np.ones(shape=(N[0],N[1]), dtype=np.bool)
        for i in range(0, N[0]):
            for h in range(int(point1[i]), int(point2[i])):
                mask[i,h] = 0
                
        offspring = np.logical_or(np.logical_and(parents[0], mask), np.logical_and(parents[1], np.logical_not(mask)))

        return offspring
    
    
    def bitFlip(self, offspring):
        rng = self.seed
        size=np.shape(offspring)
        mask = rng.choice([True, False], size = size, p = [self.mutationrate, \
                          1-self.mutationrate])
        offspring[mask] = np.logical_not(offspring[mask])
        return offspring   
    
    def __crossover(self, parents):
        if self.crossoverOp == '2P':
            return self.TwoPoint_crossover(parents)
        else:
            if self.crossoverOp == 'uniform':
                return self.uniform_crossover(parents)
            else:
                raise ValueError("Please select a valid crossover strategy")
    
    def __mutation(self, offspring):
        if self.mutationOp == 'bitFlip':
            return self.bitFlip(offspring)
        else:
            raise ValueError("Please select a valid mutation strategy")
    
    def __reproduce(self, parents):
        return self.__mutation(self.__crossover(parents))
    
    def __survivalSelection(self, parents, fparents, offspring, foffspring):
        mergedPop = np.vstack((parents, offspring))
        mergedFit = np.hstack((fparents, foffspring))
         
        idx = np.argsort(mergedFit)[0:self.popsize]
        print("Top 10 fitness: " + str(mergedFit[idx][0:10]))
        return mergedPop[idx,:], mergedFit[idx]

     
    def save(self,pop): ##Guardar la population para continuar despues
        np.savetxt('population.csv', pop, delimiter=',') 
        print('The current population is saved.')
    
    def solve(self):
        rng = self.seed
        
        numVars = self.movimientos * 3
        
        pop = rng.choice(a=[False, True], size=(self.popsize,numVars), p=[0.5, 0.5])

        
        self.env = retro.make(game='SuperMarioBros-Nes', state='Level4-1')#, record=True)
        self.action = [0,0,0,0,0,0,0,0,0,0,0,0]
        fitness = self.getFitness(pop) * -1
        
        
        currentGeneration = 0
        while currentGeneration < self.generations + 1:
            parents = self.tournament(pop, fitness)
            offspring = self.__reproduce(parents)
            foffspring = self.getFitness(offspring) * -1
            
            pop, fitness = self.__survivalSelection(pop, fitness, offspring,
                                                    foffspring)
            if(keyboard.is_pressed('s')):
              self.save(pop) 
            
            print('The best value in generation '+str(currentGeneration)+ ' is '+str(np.min(fitness)))
            if(currentGeneration%2 == 0):
                print("render of best individual")
                idxBest = np.argmin(fitness)
                self.Render_movement(pop[idxBest])
            currentGeneration += 1
        
        idxBest = np.argmin(fitness)
        self.env.close()
        return (pop[idxBest,:], fitness[idxBest])
    
    def getFitness(self, individuals):
        done = False
        fitness = np.zeros( self.popsize )
        xscroll=0
        time = 0
        scoreHi = 0
        #print("Shape of individuals: " + str(np.shape(individuals)) + "  Size of pop: " + str(self.popsize))
        
        for i in range(0, self.popsize):
            self.env.reset()
            for offset in range(0, self.movimientos):
                #self.env.render()    
                #action = env.action_space.sample()
                self.action[6] = individuals[i, offset*3 + 0];
                self.action[7] = individuals[i, offset*3 + 1];
                self.action[8] = individuals[i, offset*3 + 2];
                
                ob, rew, done, info = self.env.step(self.action)
                
                
                for d in range(0, self.dilatacion -1):    
                    ob, rew, done, info = self.env.step(self.action)
                
                if(info["lives"] == 2):   
                    xscroll = info["xscrollLo"]
                    scoreHi = info["xscrollHi"]
                
               # if(info["time"]<400 and info["lives"] == 2):  #Resuelve el problema de division entre 0
               #     VelocityOfDeath = int(xscroll)/(400-int(info["time"]))
                

                    
               # fitness[i] = VelocityOfDeath*10                    
                if(info["lives"] <= 1):
                    fitness[i] = -150
                    time = info["time"]
                    print("Me mori")
                    break
                if(done):
                    fitness[i] = info["time"]
                    print("Goal")
                    break
                
                if(keyboard.is_pressed('q')):
                    break
                
            fitness[i] = fitness[i] + xscroll + 255*scoreHi + (int(xscroll)/(400-int(time)))*10
            print("X scroll: ", xscroll, "Time: ", info["time"], "Fitness: ", fitness[i])
        return fitness
    
    def Render_movement(self, movement):
        self.env.reset()
        for offset in range(0, self.movimientos):

            #action = env.action_space.sample()
            self.action[6] = movement[offset*3 + 0];
            self.action[7] = movement[offset*3 + 1];
            self.action[8] = movement[offset*3 + 2];
            
            for d in range(0, self.dilatacion):
                ob, rew, done, info = self.env.step(self.action)
                self.env.render()
            if(info["lives"] <= 1 or done):
                break
            
            if(keyboard.is_pressed('q')):
                break
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        print("Exitingg")



solution = genericEA(movimientos=112 , dilatacion=12 , generations=1000 , populationSize=20 , recombinationRate=1 , mutationRate=0.15 , seed=1)

