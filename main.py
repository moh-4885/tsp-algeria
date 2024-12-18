import csv
import math
import random
import numpy as np

class TSPGeneticAlgorithm:
    def __init__(self, cities_file, routes_file, population_size=100, 
                 max_generations=500, mutation_rate=0.01, elite_size=10):
        """
        Initialize the Genetic Algorithm for Traveling Salesman Problem
        
        :param cities_file: Path to CSV file with city coordinates
        :param routes_file: Path to CSV file with inter-city routes
        :param population_size: Number of routes in each generation
        :param max_generations: Maximum number of generations to run
        :param mutation_rate: Probability of mutation for each route
        :param elite_size: Number of best routes to preserve between generations
        """
        # Load city coordinates
        self.cities = self.load_cities(cities_file)
        
        # Load route distances (optional, for more realistic distance calculation)
        self.routes = self.load_routes(routes_file)
        
        # Algorithm parameters
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Precompute distance matrix
        self.distance_matrix = self.compute_distance_matrix()
    
    def load_cities(self, filename):
        """
        Load city coordinates from CSV
        
        :param filename: Path to CSV file
        :return: Dictionary of city coordinates
        """
        cities = {}
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cities[row['City']] = {
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude'])
                }
        return cities
    
    def load_routes(self, filename):
        """
        Load inter-city routes from CSV
        
        :param filename: Path to CSV file
        :return: Dictionary of routes between cities
        """
        routes = {}
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['City'], row['Adjacent province'])
                routes[key] = float(row['Distance'])
        return routes
    
    def haversine_distance(self, city1, city2):
        """
        Calculate the great circle distance between two cities using Haversine formula
        
        :param city1: First city coordinates
        :param city2: Second city coordinates
        :return: Distance in kilometers
        """
        R = 6371  # Earth radius in kilometers
        
        lat1, lon1 = math.radians(city1['lat']), math.radians(city1['lon'])
        lat2, lon2 = math.radians(city2['lat']), math.radians(city2['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def compute_distance_matrix(self):
        """
        Precompute distances between all cities
        
        :return: 2D matrix of distances
        """
        city_names = list(self.cities.keys())
        n = len(city_names)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.haversine_distance(
                        self.cities[city_names[i]], 
                        self.cities[city_names[j]]
                    )
        
        return matrix
    
    def create_initial_population(self):
        """
        Create initial population of random routes
        
        :return: List of routes
        """
        city_names = list(self.cities.keys())
        population = []
        
        for _ in range(self.population_size):
            route = random.sample(city_names, len(city_names))
            population.append(route)
        
        return population
    
    def calculate_total_distance(self, route):
        """
        Calculate total distance of a route
        
        :param route: List of cities in order
        :return: Total route distance
        """
        total_distance = 0
        city_names = list(self.cities.keys())
        
        for i in range(len(route)):
            current_city_idx = city_names.index(route[i])
            next_city_idx = city_names.index(route[(i + 1) % len(route)])
            total_distance += self.distance_matrix[current_city_idx][next_city_idx]
        
        return total_distance
    
    def fitness(self, population):
        """
        Calculate fitness (inverse of total distance) for each route
        
        :param population: List of routes
        :return: List of fitness scores
        """
        return [1 / (self.calculate_total_distance(route) + 1) for route in population]
    
    def selection(self, population, fitness_scores):
        """
        Select routes for next generation using roulette wheel selection
        
        :param population: Current population of routes
        :param fitness_scores: Fitness scores for each route
        :return: Selected routes
        """
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        
        return random.choices(
            population, 
            weights=selection_probs, 
            k=self.population_size - self.elite_size
        )
    
    def crossover(self, parent1, parent2):
        """
        Create offspring using order crossover
        
        :param parent1: First parent route
        :param parent2: Second parent route
        :return: Child route
        """
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None] * len(parent1)
        
        # Copy selected segment from parent1
        child[start:end+1] = parent1[start:end+1]
        
        # Fill remaining slots with parent2 cities not yet in child
        remaining = [city for city in parent2 if city not in child]
        
        j = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = remaining[j]
                j += 1
        
        return child
    
    def mutate(self, route):
        """
        Mutate route by swapping two random cities
        
        :param route: Route to potentially mutate
        :return: Mutated route
        """
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        
        return route
    
    def solve(self):
        """
        Run the genetic algorithm to find the shortest route
        
        :return: Best route and its total distance
        """
        # Initialize population
        population = self.create_initial_population()
        
        for generation in range(self.max_generations):
            # Calculate fitness
            fitness_scores = self.fitness(population)
            
            # Find best route in current generation
            best_route_idx = fitness_scores.index(max(fitness_scores))
            best_route = population[best_route_idx]
            best_distance = self.calculate_total_distance(best_route)
            
            # Print progress
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Distance = {best_distance:.2f} km")
            
            # Select routes for next generation
            selected_routes = self.selection(population, fitness_scores)
            
            # Elitism: keep best routes
            elite = sorted(
                zip(population, fitness_scores), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.elite_size]
            
            new_population = [route for route, _ in elite]
            
            # Create new generation through crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_routes, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Return best overall route
        best_fitness_idx = self.fitness(population).index(max(self.fitness(population)))
        best_route = population[best_fitness_idx]
        best_distance = self.calculate_total_distance(best_route)
        
        return best_route, best_distance

# Example usage
def main():
    tsp = TSPGeneticAlgorithm(
        cities_file='Cities.csv', 
        routes_file='AlgeriaCities.csv'
    )
    
    best_route, total_distance = tsp.solve()
    
    print("\nBest Route:")
    for city in best_route:
        print(city)
    print(f"\nTotal Distance: {total_distance:.2f} km")

if __name__ == "__main__":
    main()