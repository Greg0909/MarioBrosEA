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

def genericEA(crossover = 'uniform', mutation = 'uniform', movimientos = 50, dilatacion = 1,
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
        self.popsize = math.floor(populationSize)
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

    
    def uniform(self, offspring):
        rng = self.seed
        size=np.shape(offspring)
        mask = rng.choice([True, False], size = size, p = [self.mutationrate, \
                          1-self.mutationrate])
        offspring[mask] = np.logical_not(offspring[mask])
        return offspring
    
   
    
    
    def __crossover(self, parents):
        if self.crossoverOp == 'arithmetic':
            return self.arithmetic(parents)
        else:
            if self.crossoverOp == 'uniform':
                return self.uniform_crossover(parents)
            else:
                raise ValueError("Please select a valid crossover strategy")
    
    def __mutation(self, offspring):
        if self.mutationOp == 'uniform':
            return self.uniform(offspring)
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
                lifes = info["lives"]
                for d in range(0, self.dilatacion -1):
                    ob, rew, done, info = self.env.step(self.action)
                #print("Action ", action, "Reward ", rew, "Info ", info)
                
                #if(info["time"] < 385):
                    #self.env.render()
                if(lifes <= 1):
                    fitness[i] = -500
                    print("Me mori")
                    break
                if(done):
                    fitness[i] = info["time"]
                    print("Me donie")
                    break
                
                if(keyboard.is_pressed('q')):
                    break
            print("Fitness acumulada: " + str(fitness[i]))
            fitness[i] = fitness[i] + info["xscrollLo"]
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
            #print("Action ", action, "Reward ", rew, "Info ", info)
            if(info["lives"] <= 1 or done):
                break
            
            if(keyboard.is_pressed('q')):
                break
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        print("Exitingg")



solution = genericEA(movimientos=66 , dilatacion=12 , generations=1000 , populationSize=10 , recombinationRate=1 , mutationRate=0.1 , seed=1)

