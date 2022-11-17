# %%
"""
### Oblig 9 - Bioinspirerte metoder
"""

# %%
import math
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

# %%
@dataclass
class Genetic:
    best_fitness: int
    average_fitness: float

    def __init__(self, best, avg):
        self.best_fitness = best
        self.average_fitness = avg

# %%
print(np.random.randint(0,256,10))

# %%
class Evolution():
    def __init__(self, max = 256, size=10, base = 8):
        self.size = size
        self.base = base
        self.max = max
        self.generations = []
        self.first_gen = np.random.randint(0, max, size=size) # 10 random tall mellom 0 og 255
        self.generations.append(self.first_gen)
        self.target = np.random.randint(0, max) # Random tall mellom 0 og 256

    # for å oppnå genetisk variasjon
    def mutation(self, x: int):
        start = np.random.randint(0, math.floor(self.base/2))
        end = np.random.randint(start, math.floor(self.base))
        x_string = str(bin(x))[2:]

        if len(x_string) != self.base:
            for i in range(self.base-len(x_string)):
                x_string = '0' + x_string

        x_string = list(x_string)
        for i in range(start, end):
            x_string[i] = '1' if x_string[i] == '0' else '0'

        return int(''.join(x_string), 2)

    # Sexual reproduction,
    def crossover(self, x, y):
        rand = np.random.randint(0, 2)
        x_string = str(bin(x if rand == 0 else y))[2:]
        y_string = str(bin(x if rand == 1 else y))[2:]
        n = 0
        if len(x_string) % 2 == 0:
            n = len(x_string) / 2
        else:
            if self.fitness(x if rand == 0 else y) > self.fitness(x if rand == 1 else y):
                n = math.ceil(len(x_string) / 2)
            else:
                n = math.ceil(len(x_string) / 2)

        n = int(n)
        child = x_string[:n]+y_string[n:]
        return int(child, 2)

    # fitness funksjon, 0 for optimal fitness i dette tilfellet
    def fitness(self, x: int):
        return -np.abs(x - self.target)

    def get_new_crossover(self, generation: zip):
        new_generation = []
        best = 0
        second = 0

        for i in range(len(generation)):
            if best > generation[i][0] > second != generation[i][1]:
                second = generation[i][1]

            if generation[i][0] > best != generation[i][1]:
                second = best
                best = generation[i][1]

        for i in generation:
            new_generation.append(self.crossover(i[1], best))
            new_generation.append(self.crossover(i[1], second))

        return new_generation

    def get_new_generation(self, old_generation):
        new_generation = self.get_new_crossover(old_generation)

        for i in np.random.randint(0, self.size, size=math.floor(self.size/2)):
            new_generation[i] = self.mutation(new_generation[i])

        return new_generation

    def train(self):
        i = 0
        train_fitness = []
        while True:

            fitnesses = [self.fitness(x) for x in self.generations[i]]
            train_fitness.append(fitnesses)
            # Pick 5 best numbers
            generation = sorted(zip(fitnesses, self.generations[i]), reverse=True)[:(math.floor(self.size/2))]
            if max(fitnesses) == 0:
                # print(self.generations)
                return [Genetic(max(f), sum(f)/len(f)) for f in train_fitness], self.generations

            new_generation = self.get_new_generation(generation)
            self.generations.append(new_generation)

            i += 1

    def train_time(self):
        i = 0
        train_time = []

        for t in range(20):
            # print(i)
            start_time = time.time()
            while True:

                fitnesses = [self.fitness(x) for x in self.generations[i]]
                # 5 best
                generation = sorted(zip(fitnesses, self.generations[i]), reverse=True)[:(math.floor(self.size/2))]
                if max(fitnesses) == 0: # stoppkriterie, funnet!
                    end_time = time.time()
                    train_time.append(end_time-start_time)
                    break
        
                new_generation = self.get_new_generation(generation)
                self.generations.append(new_generation)

                i += 1

        return sum(train_time)/len(train_time)

# %%
def print_genetics(genetic, best, average, i):
    print(f"Genetic {i}: ")
    print(genetic)
    print(f"Best Fitness: {best}")
    print(f"Average Fitness: {average}")
    print("")

# %%
def task1():
    evo = Evolution()
    x_axis = []
    y_axis_best = []
    y_axis_avg = []

    genetic, genetics = evo.train()

    for i in range(len(genetics)):
        print_genetics(genetics[i], genetic[i].best_fitness, genetic[i].average_fitness, i)
        x_axis.append(i)
        y_axis_best.append(genetic[i].best_fitness)
        y_axis_avg.append(genetic[i].average_fitness)
    
    plt.plot(x_axis, y_axis_best, label="Best fitness", color="Red")
    plt.plot(x_axis, y_axis_avg, label="Average fitness", color="Blue")
    plt.xlabel("Runs")
    plt.ylabel("Red: Best, Blue: Avg")
    plt.show()
    print(f"Target: {evo.target}")
    print(f"Runs: {i}")

# %%
def task2():
    n = np.arange(8, 18)
    time  = []

    for i in range(8, 18):
        evo = Evolution(2**i, 20, i)
        time.append(evo.train_time())
        print(f"Done with: {2**i}")
        print(f"Target: {evo.target}")
    plt.xlabel("Bit length")
    plt.ylabel("Time (s)")
    plt.plot(n, time)
    plt.show()

# %%
task1()

# %%
task2()