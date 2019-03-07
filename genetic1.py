import random #imports the random library  
import numpy as np #imports a numpy library
from tqdm import tqdm

# parameters
initial_pop_size = 3000
mutation_rate = 0.15
num_generations = 100
chromosome_length = 3
num_survivors = 150

# functions

def A(): #initialization function
    global gene_pool
    gene_pool = np.linspace(0,0.2,num = 15000) #returns an array of 5000 evenly spaced numbers from -1 to 80
    global dimensions
    dimensions = (initial_pop_size, chromosome_length) #sets the shape (dimensions) of the population 
    return np.random.choice(gene_pool, size=dimensions, replace=False) 
    #returns a randomly generated sample from an array "gene_pool" with the output shape of (100,2) 
    #without replacement to not repeat the same values from the sample pool

def B(coefficients): #error function
    k = len(data) #assigns the length of the loaded array to a variable "k"
    tot = 0 #assigns 0 to the variable "tot"
    for j in range(k): #creates a loop that iterates k times
        y = coefficients[0] * x0[j] + coefficients[1]*x1[j]+coefficients[2] 
        #finds the predicted values of the response (y) by the formula: y=b1*x+b0, 
        #where b1 is the best solution for slope and b0 is the intercept
        res = y2[j] - y 
        #calculates the residuals by formula: res = observed value - predicted value 
        tot += res**2 #finds the sum of the squared residuals, which is the error of the regression line
    return tot/k #returns the error 
    
def C(): #fitness function
    fitlist = [] #creates an empty list
    for x in range(len(current_pop)):
    #creates a loop that iterates len(curren_pop) times
    #len(current_pop) is the length of the current population
        fitlist.append(np.array([x,B(current_pop[x])])) 
        #adds the array that consists of the row number and solution error from the function B to the list "fitlist"
    return np.array(fitlist) #turns the list into the array and returns it
    
def D(): #survival selection
    random_selection = np.random.choice(range(len(fitness_vector)), num_survivors//2, replace=False) 
    #generates n=num_survivors//2 random samples within the range of len(fitness_vector) without replacement
    best = np.argmin(fitness_vector[random_selection,1]) 
    #finds the index of the minimum error value within random rows generated in "random_selection" along the "fitlist"
    best_index = random_selection[best] 
    #finds the best representative with least error from the chosen samples,
    #which is its index within the fitness_vector
    return current_pop[int(fitness_vector[best_index][0])] 
    #finds the order number of the best representative from fitness_vector and returns the current best solution 

def E(): #crossover
    duplicate_size = len(new_population) - len(survivors) #calculates the size (length) of population duplicate
    duplicate_survivors = np.zeros((duplicate_size, chromosome_length)) 
    #creates a new survivor array with the shape of (duplicate_size, 2) filled with zeros 
    for x in range(chromosome_length): #creates a loop that iterates through each column in the survivor array
        duplicate_survivors[:, x] = np.repeat(survivors[:, x], 4, axis=0) #repeats elements of the array 4 times 
        #assigns new values to certain items in all rows within the array "duplicate_survivors"
        #axis along which to repeat the values is set to 0
        duplicate_survivors[:, x] = np.random.permutation(duplicate_survivors[:, x]) 
        #randomly permutes a sequence of values  
        #assigns new values to certain items in all rows within the array "duplicate_survivors"
    return duplicate_survivors #returns the array

def F(array): # mutation
    for chromosome in range(len(array)): 
        #creates a loop that goes through every single chromosome in the population
        if(random.random() < mutation_rate): 
            #creates a conditional for choosing the genes that will be mutated by the Uniform mutation method
            swapWith = np.random.choice(np.linspace(-1,80,num = 5000))
            #randomly chooses a number from newly generated list of numbers that will replace one of the genes in the chromosome
            index=int(random.random()*chromosome_length) 
            #chooses the index of the gene that will be changed, whether it is the first one or second
            array[chromosome][index]=swapWith #mutates the value of the gene
    return array #returns the array
    

########################################################################
# Start of main program
current_pop = A() #assigns the returned array from function A to the variable "current_pop"
new_population = np.zeros((num_survivors * 5, chromosome_length)) 
#creates an array with shape (num_survivors * 5, chromosome_length) filled with zeros 
#assigns it to the variable "new_population"
# main loop
for i in tqdm(range(num_generations)): #creates a loop that iterates n=num_generations times
    
    fitness_vector = C() #assigns the returned array from function C to the variable "fitness_vector"
    survivors = np.zeros((num_survivors, chromosome_length)) 
    #creates an array with shape (num_survivors, chromosome_length) filled with zeros 
    #assigns it to the variable "survivors"
    for n in range(len(survivors)): #creates a loop that iterates len(survivors) times
        survivors[n] = D() #assigns the best representative from fitness_vector to the items in the array
    new_population[:len(survivors)] = survivors 
    #changes the first len(survivors) elements of the new_population to the values from "survivors" array 
    new_population[len(survivors):] = E() 
    #assigns the array from function E as the rest of the elements in "new_population"
    
    new_population = F(new_population) #assigns the returned array to the variable 'new_population'
    
    current_pop = new_population #assigns the array from "new_population" to the "current_population"
    new_population = np.zeros((num_survivors * 5, chromosome_length)) 
    #creates an array with shape (num_survivors * 5, chromosome_length) filled with zeros 
    #assigns it to the variable "new_population"
fitness_vector = C() #assigns the returned array from function C to the variable "fitness_vector"
best_solution = current_pop[np.argmin(fitness_vector[:,1])] #finds the best solution based on the residuals 
print("The best solution is", best_solution) #prints the best solution
print("with error equal to approximately", B(best_solution)) #prints the error of the best solution
