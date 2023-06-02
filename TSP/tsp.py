import random
import math
import argparse


class TSP:
    def __init__(self):
        self.population_size = 100
        self.mutation_rate = 0.02
        self.competition_size = 5
        self.generations = 100
        self.initial_temperature = 1000
        self.cooling_rate = 0.99
        self.stopping_temperature = 0.1

    def calculate_distance(self, city1, city2):
        lat1, lon1 = city1[0], city1[1]
        lat2, lon2 = city2[0], city2[1]
        radius = 6400
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        temp_dist = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        final_dist = 2 * \
            math.atan2(math.sqrt(temp_dist), math.sqrt(1-temp_dist))
        distance = radius * final_dist
        return distance

    def calculate_total_distance(self, city, cities):
        total_distance = 0
        for idx in range(len(city)):
            from_city = cities[city[idx]]
            to_city = cities[city[(idx+1) % len(city)]]
            total_distance += self.calculate_distance(from_city, to_city)
        return total_distance

    def generate_initial_solution(self, cities):
        city_names = list(cities.keys())
        return random.sample(city_names, len(city_names))

    def acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / temperature)

    def neighbor(self, solution):
        new_solution = solution.copy()
        index1 = random.randint(0, len(solution) - 1)
        index2 = random.randint(0, len(solution) - 1)
        new_solution[index1], new_solution[index2] = new_solution[index2], new_solution[index1]
        return new_solution

    def create_initial_population(self, city_names, population_size):
        population = []
        for _ in range(population_size):
            random.shuffle(city_names)
            population.append(city_names.copy())
        return population

    def cities_selection(self, population, competition_size, cities):
        competition = random.sample(population, competition_size)
        competition.sort(
            key=lambda city: self.calculate_total_distance(city, cities))
        return competition[0]

    def cities_crossover(self, parent1, parent2):
        start_index = random.randint(0, len(parent1) - 1)
        end_index = random.randint(start_index + 1, len(parent1))
        child = parent1[start_index:end_index]
        for city in parent2:
            if city not in child:
                child.append(city)
        return child

    def mutate_route(self, city, mutation_rate):
        for idx in range(len(city)):
            if random.random() < mutation_rate:
                rand = random.randint(0, len(city) - 1)
                city[idx], city[rand] = city[rand], city[idx]
        return city

    def evolve_population(self, population, cities):
        new_population = []
        temp_pop = sorted(
            population, key=lambda city: self.calculate_total_distance(city, cities))[:1]
        new_population.extend(temp_pop)
        while len(new_population) < self.population_size:
            parent1 = self.cities_selection(
                population, self.competition_size, cities)
            parent2 = self.cities_selection(
                population, self.competition_size, cities)
            child = self.cities_crossover(parent1, parent2)
            child = self.mutate_route(child, self.mutation_rate)
            new_population.append(child)
        return new_population

    def hill_climbing(self, cities):
        city_names = list(cities.keys())
        num_cities = len(city_names)
        best_order = list(city_names)
        random.shuffle(best_order)
        best_distance = self.calculate_total_distance(best_order, cities)

        while True:
            has_change = False
            for idx in range(num_cities):
                for j in range(idx + 1, num_cities):
                    new_order = best_order.copy()
                    new_order[idx], new_order[j] = new_order[j], new_order[idx]
                    new_distance = self.calculate_total_distance(
                        new_order, cities)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_order = new_order
                        has_change = True
            if not has_change:
                break
        return best_order, best_distance

    def simulated_annealing(self, cities):
        current_solution = self.generate_initial_solution(cities)
        best_solution = current_solution
        current_distance = self.calculate_total_distance(
            current_solution, cities)
        best_distance = current_distance
        temperature = self.initial_temperature

        while temperature > self.stopping_temperature:
            new_solution = self.neighbor(current_solution)
            new_cost = self.calculate_total_distance(new_solution, cities)
            ap = self.acceptance_probability(
                current_distance, new_cost, temperature)
            if ap > random.random():
                current_solution = new_solution
                current_distance = new_cost
            if new_cost < best_distance:
                best_solution = new_solution
                best_distance = new_cost
            temperature *= self.cooling_rate
        return best_solution, best_distance

    def genetic_algorithm(self, city_names, cities):
        population = self.create_initial_population(
            city_names, self.population_size)
        best_distance = math.inf
        best_route = None
        for _ in range(self.generations):
            population = self.evolve_population(population, cities)
            current_best_route = min(
                population, key=lambda city: self.calculate_total_distance(city, cities))
            current_best_distance = self.calculate_total_distance(
                current_best_route, cities)
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_route = current_best_route
        return best_route, best_distance


def main():

    def parse_city_file(file_path):
        cities = {}
        with open(file_path, 'r') as file:
            for line in file:
                name, lat, lon = line.strip().split(',')
                cities[name.strip()] = (float(lat), float(lon))
        return cities

    def generate_sample(num_sample):
        city_items = list(parse_city_file(args.file).items())
        cities = dict(random.sample(city_items, num_sample))
        city_names = list(cities.keys())
        return cities, city_names

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", choices=["ga", "hc", "sa"], default="ga", help="Algorithm to use (default: ga)")
    parser.add_argument("--file", required=True,)
    args = parser.parse_args()

    cities, city_names = generate_sample(20)
    tsp = TSP()

    if args.algorithm == "hc":
        best_order, best_distance = tsp.hill_climbing(cities)
        print("Best Route (Hill Climbing):", best_order)
        print("Best Distance:", best_distance)

    elif args.algorithm == "sa":
        best_solution, best_distance = tsp.simulated_annealing(cities)
        print("Best Route (Simulated Annealing):", best_solution)
        print("Best Distance:", best_distance)

    elif args.algorithm == "ga":
        best_route, best_distance = tsp.genetic_algorithm(city_names, cities)
        print("Best Route (Genetic Algorithm):", best_route)
        print("Best Distance:", best_distance)
    else:
        print("Invalid algorithm.")
        return


if __name__ == "__main__":
    main()
