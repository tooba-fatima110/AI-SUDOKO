# import numpy
# import random

# random.seed()

# Nd = 9  # Number of digits (in the case of standard Sudoku puzzles, this is 9).


# class Population(object):
#     """ A set of candidate solutions to the Sudoku puzzle. These candidates are also known as the chromosomes in the population. """

#     def __init__(self):
#         self.candidates = []

#     def seed(self, Nc, given):
#         self.candidates = []

#         # Determine the legal values that each square can take.
#         helper = Candidate()
#         helper.values = [[[] for j in range(0, Nd)] for i in range(0, Nd)]
#         for row in range(0, Nd):
#             for column in range(0, Nd):
#                 for value in range(1, 10):
#                     if (given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or
#                                                                 given.is_block_duplicate(row, column, value) or
#                                                                 given.is_row_duplicate(row, value)):
#                         # Value is available.
#                         helper.values[row][column].append(value)
#                     elif (given.values[row][column] != 0):
#                         # Given/known value from file.
#                         helper.values[row][column].append(given.values[row][column])
#                         break

#         # Seed a new population.
#         for p in range(Nc):
#             g = Candidate()
#             for i in range(Nd):  # New row in candidate.
#                 row = numpy.zeros(Nd)

#                 # Fill in the givens.
#                 for j in range(Nd):  # New column j value in row i.

#                     # If value is already given, don't change it.
#                     if given.values[i][j] != 0:
#                         row[j] = given.values[i][j]
#                     # Fill in the gaps using the helper board.
#                     elif given.values[i][j] == 0:
#                         row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

#                 # If we don't have a valid board, then try again. There must be no duplicates in the row.
#                 while len(list(set(row))) != Nd:
#                     for j in range(Nd):
#                         if given.values[i][j] == 0:
#                             row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

#                 g.values[i] = row

#             self.candidates.append(g)

#         # Compute the fitness of all candidates in the population.
#         self.update_fitness()

#         print("Seeding complete.")
#         return

#     def update_fitness(self):
#         """ Update fitness of every candidate/chromosome. """
#         for candidate in self.candidates:
#             candidate.update_fitness()
#         return

#     def sort(self):
#         # Sort candidates by fitness in descending order
#         self.candidates = sorted(self.candidates, key=self.sort_fitness, reverse=True)

#     def sort_fitness(self, candidate):
#         # Return the fitness score for sorting
#         return candidate.fitness  # Assuming candidates have a 'fitness' attribute


# class Candidate(object):
#     """ A candidate solution to the Sudoku puzzle. """

#     def __init__(self):
#         self.values = numpy.zeros((Nd, Nd), dtype=int)
#         self.fitness = None

#     def update_fitness(self):
#         """ The fitness of a candidate solution is determined by how close it is to being the actual solution to the puzzle. """
#         row_count = numpy.zeros(Nd)
#         column_count = numpy.zeros(Nd)
#         block_count = numpy.zeros(Nd)
#         row_sum = 0
#         column_sum = 0
#         block_sum = 0

#         for i in range(Nd):  # For each row...
#             for j in range(Nd):  # For each number within it...
#                 if self.values[i][j] != 0:
#                     row_count[self.values[i][j] - 1] += 1  # Update list with occurrence of a particular number.

#             row_sum += (1.0 / len(set(row_count))) / Nd
#             row_count = numpy.zeros(Nd)

#         for i in range(Nd):  # For each column...
#             for j in range(Nd):  # For each number within it...
#                 if self.values[j][i] != 0:
#                     column_count[self.values[j][i] - 1] += 1  # Update list with occurrence of a particular number.

#             column_sum += (1.0 / len(set(column_count))) / Nd
#             column_count = numpy.zeros(Nd)

#         # For each block...
#         for i in range(0, Nd, 3):
#             for j in range(0, Nd, 3):
#                 for k in range(3):
#                     for l in range(3):
#                         if self.values[i + k][j + l] != 0:
#                             block_count[self.values[i + k][j + l] - 1] += 1

#                 block_sum += (1.0 / len(set(block_count))) / Nd
#                 block_count = numpy.zeros(Nd)

#         # Calculate overall fitness.
#         if (int(row_sum) == 1 and int(column_sum) == 1 and int(block_sum) == 1):
#             fitness = 1.0
#         else:
#             fitness = column_sum * block_sum

#         self.fitness = fitness
#         return

#     def mutate(self, mutation_rate, given):
#         """ Mutate a candidate by picking a row, and then picking two values within that row to swap. """
#         r = random.uniform(0, 1)
#         success = False
#         if r < mutation_rate:  # Mutate.
#             while not success:
#                 row1 = random.randint(0, Nd - 1)
#                 row2 = random.randint(0, Nd - 1)
#                 while row2 == row1:  # Ensure we are picking different rows.
#                     row2 = random.randint(0, Nd - 1)

#                 from_column = random.randint(0, Nd - 1)
#                 to_column = random.randint(0, Nd - 1)
#                 while from_column == to_column:
#                     to_column = random.randint(0, Nd - 1)

#                 # Check if the two places are free...
#                 if (given.values[row1][from_column] == 0 and
#                         given.values[row1][to_column] == 0):
#                     # Check for duplicates
#                     if (not given.is_column_duplicate(to_column, self.values[row1][from_column]) and
#                             not given.is_column_duplicate(from_column, self.values[row2][to_column]) and
#                             not given.is_block_duplicate(row2, to_column, self.values[row1][from_column]) and
#                             not given.is_block_duplicate(row1, from_column, self.values[row2][to_column])):
#                         # Swap values.
#                         temp = self.values[row2][to_column]
#                         self.values[row2][to_column] = self.values[row1][from_column]
#                         self.values[row1][from_column] = temp
#                         success = True

#         return success


# class Given(Candidate):
#     """ The grid containing the given/known values. """

#     def __init__(self, values):
#         self.values = values

#     def is_row_duplicate(self, row, value):
#         """ Check whether there is a duplicate of a fixed/given value in a row. """
#         for column in range(Nd):
#             if self.values[row][column] == value:
#                 return True
#         return False

#     def is_column_duplicate(self, column, value):
#         """ Check whether there is a duplicate of a fixed/given value in a column. """
#         for row in range(Nd):
#             if self.values[row][column] == value:
#                 return True
#         return False

#     def is_block_duplicate(self, row, column, value):
#         """ Check whether there is a duplicate of a fixed/given value in a 3 x 3 block. """
#         i = 3 * (row // 3)
#         j = 3 * (column // 3)

#         for k in range(3):
#             for l in range(3):
#                 if self.values[i + k][j + l] == value:
#                     return True
#         return False


# class Tournament(object):
#     """ The crossover function requires two parents to be selected from the population pool. The Tournament class is used to do this. """

#     def __init__(self):
#         pass

#     def compete(self, candidates):
#         """ Pick 2 random candidates from the population and get them to compete against each other. """
#         c1 = candidates[random.randint(0, len(candidates) - 1)]
#         c2 = candidates[random.randint(0, len(candidates) - 1)]
#         f1 = c1.fitness
#         f2 = c2.fitness

#         # Find the fittest and the weakest.
#         if f1 > f2:
#             return c1
#         else:
#             return c2

#     def select(self, population):
#         """ Return the best two candidates from the population. """
#         return [self.compete(population.candidates) for _ in range(2)]


# def solve(given, Nc=100, Ne=1000, mutation_rate=0.05):
#     """ Solves the Sudoku puzzle using a genetic algorithm. """
#     pop = Population()
#     pop.seed(Nc, given)

#     for _ in range(Ne):  # Number of generations.
#         pop.sort()
#         if pop.candidates[0].fitness == 1.0:  # Stop if the first candidate is a perfect solution.
#             print("Solution found.")
#             return pop.candidates[0]

#         new_candidates = []
#         tournament = Tournament()
#         for _ in range(Nc):
#             parents = tournament.select(pop)
#             offspring = Candidate()

#             # Cross over
#             for i in range(Nd):
#                 crossover_row = random.choice([parents[0].values[i], parents[1].values[i]])
#                 offspring.values[i] = crossover_row

#             new_candidates.append(offspring)

#         for candidate in new_candidates:
#             candidate.mutate(mutation_rate, given)

#         pop.candidates = new_candidates
#         pop.update_fitness()

#     print("No solution found.")
#     return None


# # Example of how to initialize a Given instance and solve a Sudoku puzzle.
# if __name__ == "__main__":
#     # Initialize a Sudoku puzzle with zeros (representing empty cells).
#     example_sudoku = numpy.zeros((Nd, Nd), dtype=int)
#     # Fill in some example values.
#     example_sudoku[0][0] = 5
#     example_sudoku[1][1] = 6
#     example_sudoku[4][4] = 1
#     example_sudoku[5][5] = 9

#     # Initialize the Given puzzle.
#     given = Given(example_sudoku)
    
#     # Solve the puzzle.
#     solution = solve(given)
#     if solution:
#         print("Solved Sudoku:")
#         print(solution.values)
import random as rndm
import time

# Part 1: Defining Genes and Chromosomes
def make_gene(initial=None):
    if initial is None:
        initial = [0] * 9
    mapp = {}
    gene = list(range(1, 10))
    rndm.shuffle(gene)
    for i in range(9):
        mapp[gene[i]] = i
    for i in range(9):
        if initial[i] != 0 and gene[i] != initial[i]:
            temp = gene[i], gene[mapp[initial[i]]]
            gene[mapp[initial[i]]], gene[i] = temp
            mapp[initial[i]], mapp[temp[0]] = i, mapp[initial[i]]
    return gene

def make_chromosome(initial=None):
    if initial is None:
        initial = [[0] * 9] * 9
    chromosome = []
    for i in range(9):
        chromosome.append(make_gene(initial[i]))
    return chromosome

# Part 2: Making First Generation
def make_population(count, initial=None):
    if initial is None:
        initial = [[0] * 9] * 9
    population = []
    for _ in range(count):
        population.append(make_chromosome(initial))
    return population

# Part 3: Fitness Function
def get_fitness(chromosome):
    """Calculate the fitness of a chromosome (puzzle)."""
    fitness = 0

    # Row fitness
    for row in chromosome:
        seen = {}
        for num in row:
            if num in seen:
                seen[num] += 1
            else:
                seen[num] = 1
        for key in seen:
            fitness -= (seen[key] - 1)

    # Column fitness
    for i in range(9):
        seen = {}
        for j in range(9):
            num = chromosome[j][i]
            if num in seen:
                seen[num] += 1
            else:
                seen[num] = 1
        for key in seen:
            fitness -= (seen[key] - 1)

    # Subgrid fitness
    for m in range(3):
        for n in range(3):
            seen = {}
            for i in range(3 * m, 3 * (m + 1)):
                for j in range(3 * n, 3 * (n + 1)):
                    num = chromosome[i][j]
                    if num in seen:
                        seen[num] += 1
                    else:
                        seen[num] = 1
            for key in seen:
                fitness -= (seen[key] - 1)

    return fitness

# def get_fitness(chromosome):
#     """Calculate the fitness of a chromosome (puzzle)."""
#     fitness = 0
#     for i in range(9):  # For each column
#         seen = {}
#         for j in range(9):  # Check each cell in the column
#             if chromosome[j][i] in seen:
#                 seen[chromosome[j][i]] += 1
#             else:
#                 seen[chromosome[j][i]] = 1
#         for key in seen:  # Subtract fitness for repeated numbers
#             fitness -= (seen[key] - 1)

#     for m in range(3):  # For each 3x3 square
#         for n in range(3):
#             seen = {}
#             for i in range(3 * n, 3 * (n + 1)):  # Check cells in 3x3 square
#                 for j in range(3 * m, 3 * (m + 1)):
#                     if chromosome[j][i] in seen:
#                         seen[chromosome[j][i]] += 1
#                     else:
#                         seen[chromosome[j][i]] = 1
#             for key in seen:  # Subtract fitness for repeated numbers
#                 fitness -= (seen[key] - 1)
#     return fitness

def pch(ch):
    for i in range(9):
        if i % 3 == 0 and i != 0:  # Print a horizontal line after every 3 rows
            print("---------------------")
        for j in range(9):
            if j % 3 == 0 and j != 0:  # Print a vertical line after every 3 columns
                print("|", end=" ")
            print(ch[i][j], end=" ")
        print("")  # Newline after each row


# Part 4: Crossover and Mutation
def crossover(ch1, ch2):
    new_child_1 = []
    new_child_2 = []
    for i in range(9):
        x = rndm.randint(0, 1)
        if x == 1:
            new_child_1.append(ch1[i])
            new_child_2.append(ch2[i])
        else:
            new_child_1.append(ch2[i])
            new_child_2.append(ch1[i])
    return new_child_1, new_child_2

def mutation(ch, pm, initial):
    for i in range(9):
        x = rndm.randint(0, 100)
        if x < pm * 100:
            ch[i] = make_gene(initial[i])
    return ch

# Part 5: Implementing The Genetic Algorithm
def read_puzzle(address):
    puzzle = []
    with open(address, 'r') as f:
        for row in f:
            temp = row.split()
            puzzle.append([int(c) for c in temp])
    return puzzle

def r_get_mating_pool(population):
    fitness_list = []
    pool = []
    for chromosome in population:
        fitness = get_fitness(chromosome)
        fitness_list.append((fitness, chromosome))
    fitness_list.sort()
    weight = list(range(1, len(fitness_list) + 1))
    for _ in range(len(population)):
        ch = rndm.choices(fitness_list, weight)[0]
        pool.append(ch[1])
    return pool

def get_offsprings(population, initial, pm, pc):
    new_pool = []
    i = 0
    while i < len(population):
        ch1 = population[i]
        ch2 = population[(i + 1) % len(population)]
        x = rndm.randint(0, 100)
        if x < pc * 100:
            ch1, ch2 = crossover(ch1, ch2)
        new_pool.append(mutation(ch1, pm, initial))
        new_pool.append(mutation(ch2, pm, initial))
        i += 2
    return new_pool

# Population size
POPULATION = 500  #2000

# Number of generations
REPETITION = 500   #1000

# Probability of mutation
PM = 0.1     #0.1

# Probability of crossover
PC = 0.7     #0.95

# Main genetic algorithm function

def genetic_algorithm(initial_file):
    initial = read_puzzle(initial_file)
    population = make_population(POPULATION, initial)
    generation=0
    for _ in range(REPETITION):
        generation+=1
        mating_pool = r_get_mating_pool(population)
        rndm.shuffle(mating_pool)
        population = get_offsprings(mating_pool, initial, PM, PC)
        fit = [get_fitness(c) for c in population]
        m = max(fit)
        print(f"Generation {generation}: Best Fitness = {m}")
        if m == 0:
            return population
    return population
# def genetic_algorithm(initial_file):
#     initial = read_puzzle(initial_file)
#     population = make_population(POPULATION, initial)
#     generation = 0
#     max_fitness_repeats = 20  # Number of generations with the same fitness value before stopping
#     last_max_fitness = None  # Track the last maximum fitness
#     repeat_count = 0  # Counter for repeated fitness values

#     for _ in range(REPETITION):
#         generation += 1
#         mating_pool = r_get_mating_pool(population)
#         rndm.shuffle(mating_pool)
#         population = get_offsprings(mating_pool, initial, PM, PC)
#         fit = [get_fitness(c) for c in population]
#         max_fitness = max(fit)
        
#         # Print generation and fitness information
#         print(f"Generation {generation}: Best Fitness = {max_fitness}")

#         # Check if the maximum fitness has remained the same
#         if max_fitness == last_max_fitness:
#             repeat_count += 1
#         else:
#             repeat_count = 0  # Reset the counter if fitness changes
#             last_max_fitness = max_fitness  # Update the last max fitness
        
#         # Stopping conditions
#         if max_fitness == 0:  # Found a solution with perfect fitness
#             print("Solution found!")
#             return population
#         elif repeat_count >= max_fitness_repeats:  # No improvement for specified generations
#             print("Stopping early due to fitness plateau.")
#             return population
        
#     # If the loop ends without meeting stopping conditions
#     return population


# Run the algorithm and time it
if __name__ == "__main__":
    tic = time.time()
    r = genetic_algorithm("puzzle_mild.txt")  # Adjust this path as necessary
    toc = time.time()
    print("time_taken: ", toc - tic)
    fit = [get_fitness(c) for c in r]
    m = max(fit)

    # Print the chromosome with the highest fitness
    for c in r:
        if get_fitness(c) == m:
            pch(c)
            break


# def genetic_algorithm(initial_file):
#     initial = read_puzzle(initial_file)
#     population = make_population(POPULATION, initial)
#     generation=0
#     for _ in range(REPETITION):
#         generation+=1
#         mating_pool = r_get_mating_pool(population)
#         rndm.shuffle(mating_pool)
#         population = get_offsprings(mating_pool, initial, PM, PC)
#         fit = [get_fitness(c) for c in population]
#         m = max(fit)
#         print(f"Generation {generation}: Best Fitness = {m}")
#         if m == 0:
#             return population
#     return population