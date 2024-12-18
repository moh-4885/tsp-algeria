import streamlit as st
import csv
import math
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

class TSPGeneticAlgorithm:
    def __init__(self, cities, population_size=100, 
                 max_generations=500, mutation_rate=0.01, elite_size=10):
        """
        Initialize the Genetic Algorithm for Traveling Salesman Problem
        
        :param cities: DataFrame with city coordinates
        :param population_size: Number of routes in each generation
        :param max_generations: Maximum number of generations to run
        :param mutation_rate: Probability of mutation for each route
        :param elite_size: Number of best routes to preserve between generations
        """
        # Algorithm parameters
        self.cities = cities
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Precompute distance matrix
        self.distance_matrix = self.compute_distance_matrix()
    
    def haversine_distance(self, city1, city2):
        """
        Calculate the great circle distance between two cities using Haversine formula
        
        :param city1: First city coordinates
        :param city2: Second city coordinates
        :return: Distance in kilometers
        """
        R = 6371  # Earth radius in kilometers
        
        lat1, lon1 = math.radians(city1['Latitude']), math.radians(city1['Longitude'])
        lat2, lon2 = math.radians(city2['Latitude']), math.radians(city2['Longitude'])
        
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
        n = len(self.cities)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.haversine_distance(
                        self.cities.iloc[i], 
                        self.cities.iloc[j]
                    )
        
        return matrix
    
    def create_initial_population(self):
        """
        Create initial population of random routes
        
        :return: List of routes
        """
        city_names = list(self.cities['City'])
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
        city_names = list(self.cities['City'])
        
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
    
    def solve(self, progress_bar=None):
        """
        Run the genetic algorithm to find the shortest route
        
        :param progress_bar: Optional Streamlit progress bar
        :return: Best route and its total distance
        """
        # Initialize population
        population = self.create_initial_population()
        
        best_routes = []
        best_distances = []
        
        for generation in range(self.max_generations):
            # Calculate fitness
            fitness_scores = self.fitness(population)
            
            # Find best route in current generation
            best_route_idx = fitness_scores.index(max(fitness_scores))
            best_route = population[best_route_idx]
            best_distance = self.calculate_total_distance(best_route)
            
            # Store best route for each generation
            best_routes.append(best_route)
            best_distances.append(best_distance)
            
            # Update progress bar if provided
            if progress_bar:
                progress_bar.progress((generation + 1) / self.max_generations)
            
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
        
        return best_route, best_distance, best_routes, best_distances

def load_cities(filename):
    """
    Load city coordinates from CSV
    
    :param filename: Path to CSV file
    :return: DataFrame of cities
    """
    return pd.read_csv(filename)

def create_route_map(cities, route):
    """
    Create a Plotly map of the route
    
    :param cities: DataFrame of cities
    :param route: List of cities in order
    :return: Plotly figure
    """
    # Filter cities in the route
    route_cities = cities[cities['City'].isin(route)]
    
    # Create lines connecting cities
    route_lines = []
    for i in range(len(route)):
        city1 = route_cities[route_cities['City'] == route[i]].iloc[0]
        city2 = route_cities[route_cities['City'] == route[(i + 1) % len(route)]].iloc[0]
        
        route_lines.append(
            go.Scattermapbox(
                mode='lines',
                lon=[city1['Longitude'], city2['Longitude']],
                lat=[city1['Latitude'], city2['Latitude']],
                line=dict(width=2, color='red')
            )
        )
    
    # City markers
    route_markers = go.Scattermapbox(
        mode='markers+text',
        lon=route_cities['Longitude'],
        lat=route_cities['Latitude'],
        text=route_cities['City'],
        textposition='bottom center',
        marker=dict(size=10, color='blue')
    )
    
    # Create map layout
    layout = go.Layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(
                lat=route_cities['Latitude'].mean(),
                lon=route_cities['Longitude'].mean()
            ),
            zoom=4
        ),
        showlegend=False,
        height=600
    )
    
    # Create figure
    fig = go.Figure(data=route_lines + [route_markers], layout=layout)
    
    return fig

def main():
    st.title('Genetic Algorithm: Traveling Salesman Problem in Algeria')
    
    # Load cities
    cities = load_cities('Cities.csv')
    
    # Sidebar for algorithm parameters
    st.sidebar.header('Algorithm Parameters')
    population_size = st.sidebar.slider(
        'Population Size', 
        min_value=50, 
        max_value=500, 
        value=100
    )
    max_generations = st.sidebar.slider(
        'Max Generations', 
        min_value=100, 
        max_value=1000, 
        value=500
    )
    mutation_rate = st.sidebar.slider(
        'Mutation Rate', 
        min_value=0.001, 
        max_value=0.1, 
        value=0.01,
        step=0.001
    )
    elite_size = st.sidebar.slider(
        'Elite Size', 
        min_value=1, 
        max_value=20, 
        value=10
    )
    
    # Run algorithm button
    if st.button('Find Optimal Route'):
        # Progress bar
        progress_bar = st.progress(0)
        
        # Create and solve TSP
        tsp = TSPGeneticAlgorithm(
            cities, 
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
            elite_size=elite_size
        )
        
        # Solve the problem
        best_route, total_distance, best_routes, best_distances = tsp.solve(progress_bar)
        
        # Display results
        st.subheader('Optimal Route')
        st.write('Route Order:')
        st.write(best_route)
        st.write(f'Total Distance: {total_distance:.2f} km')
        
        # Create and display route map
        route_map = create_route_map(cities, best_route)
        st.plotly_chart(route_map)
        
        # Plot convergence
        st.subheader('Optimization Convergence')
        conv_df = pd.DataFrame({
            'Generation': range(len(best_distances)),
            'Best Distance': best_distances
        })
        fig = px.line(
            conv_df, 
            x='Generation', 
            y='Best Distance', 
            title='Route Distance Over Generations'
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()