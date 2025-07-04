import numpy as np
import random
import copy
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    genes: List[float]  # Refactoring parameters
    objectives: List[float]  # Objective function values
    fitness: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0
    refactored_code: str = ""
    pattern_type: str = ""
    
class NSGA2:
    """Non-dominated Sorting Genetic Algorithm II for multi-objective optimization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.population_size = config["genetic_algorithm"]["nsga2"]["population_size"]
        self.generations = config["genetic_algorithm"]["nsga2"]["generations"]
        self.crossover_prob = config["genetic_algorithm"]["nsga2"]["crossover_probability"]
        self.mutation_prob = config["genetic_algorithm"]["nsga2"]["mutation_probability"]
        self.tournament_size = config["genetic_algorithm"]["nsga2"]["tournament_size"]
        self.elite_size = config["genetic_algorithm"]["nsga2"]["elite_size"]
        
        # Objective weights
        self.objectives = config["genetic_algorithm"]["objectives"]
        self.num_objectives = len(self.objectives)
        
        # Gene bounds (refactoring parameters)
        self.gene_bounds = self._initialize_gene_bounds()
        self.num_genes = len(self.gene_bounds)
        
        # Statistics tracking
        self.generation_stats = []
        self.pareto_front_history = []
        
    def _initialize_gene_bounds(self) -> List[Tuple[float, float]]:
        """Initialize bounds for refactoring parameters."""
        return [
            (0.0, 1.0),  # Extract method threshold
            (0.0, 1.0),  # Complexity reduction weight
            (0.0, 1.0),  # Readability importance
            (0.0, 1.0),  # Performance weight
            (0.0, 1.0),  # Maintainability weight
            (0.0, 1.0),  # Pattern confidence threshold
            (0.0, 1.0),  # Code similarity threshold
            (0.0, 1.0),  # Refactoring aggressiveness
        ]
    
    def initialize_population(self) -> List[Individual]:
        """Initialize the population randomly."""
        population = []
        
        for _ in range(self.population_size):
            genes = []
            for lower, upper in self.gene_bounds:
                genes.append(random.uniform(lower, upper))
            
            individual = Individual(
                genes=genes,
                objectives=[0.0] * self.num_objectives
            )
            population.append(individual)
        
        logger.info(f"Initialized population of {self.population_size} individuals")
        return population
    
    def evaluate_population(
        self,
        population: List[Individual],
        objective_functions: List[Callable],
        code_analyzer
    ) -> List[Individual]:
        """Evaluate objectives for all individuals in population."""
        
        for individual in population:
            objectives = []
            
            for obj_func in objective_functions:
                obj_value = obj_func(individual, code_analyzer)
                objectives.append(obj_value)
            
            individual.objectives = objectives
            
            # Calculate weighted fitness
            individual.fitness = sum(
                weight * obj_val for weight, obj_val in 
                zip([obj["weight"] for obj in self.objectives], objectives)
            )
        
        return population
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Perform fast non-dominated sorting."""
        fronts = []
        dominated_solutions = {i: [] for i in range(len(population))}
        domination_counts = {i: 0 for i in range(len(population))}
        
        # Find domination relationships
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if self._dominates(p, q):
                        dominated_solutions[i].append(j)
                    elif self._dominates(q, p):
                        domination_counts[i] += 1
        
        # Find first front
        first_front = []
        for i in range(len(population)):
            if domination_counts[i] == 0:
                population[i].rank = 0
                first_front.append(i)
        
        fronts.append(first_front)
        
        # Find subsequent fronts
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        # Convert indices to individuals
        front_individuals = []
        for front in fronts:
            if front:  # Only add non-empty fronts
                front_individuals.append([population[i] for i in front])
        
        return front_individuals
    
    def _dominates(self, p: Individual, q: Individual) -> bool:
        """Check if individual p dominates individual q."""
        at_least_one_better = False
        
        for i, obj in enumerate(self.objectives):
            if obj["maximize"]:
                if p.objectives[i] < q.objectives[i]:
                    return False
                elif p.objectives[i] > q.objectives[i]:
                    at_least_one_better = True
            else:
                if p.objectives[i] > q.objectives[i]:
                    return False
                elif p.objectives[i] < q.objectives[i]:
                    at_least_one_better = True
        
        return at_least_one_better
    
    def calculate_crowding_distance(self, front: List[Individual]) -> List[Individual]:
        """Calculate crowding distance for individuals in a front."""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return front
        
        # Initialize crowding distance
        for individual in front:
            individual.crowding_distance = 0.0
        
        # Calculate crowding distance for each objective
        for m in range(self.num_objectives):
            # Sort by objective m
            front.sort(key=lambda x: x.objectives[m])
            
            # Set boundary points to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distance for intermediate points
            if front[-1].objectives[m] - front[0].objectives[m] > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i+1].objectives[m] - front[i-1].objectives[m]) / \
                              (front[-1].objectives[m] - front[0].objectives[m])
                    front[i].crowding_distance += distance
        
        return front
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Perform tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        
        # Select best individual based on rank and crowding distance
        best = tournament[0]
        for individual in tournament[1:]:
            if (individual.rank < best.rank or 
                (individual.rank == best.rank and 
                 individual.crowding_distance > best.crowding_distance)):
                best = individual
        
        return copy.deepcopy(best)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform simulated binary crossover (SBX)."""
        if random.random() > self.crossover_prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        eta_c = 20  # Distribution index for crossover
        
        child1_genes = []
        child2_genes = []
        
        for i in range(self.num_genes):
            if random.random() <= 0.5:
                # Perform crossover
                y1, y2 = parent1.genes[i], parent2.genes[i]
                
                if abs(y1 - y2) > 1e-14:
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    # Calculate beta
                    rand = random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1 / (eta_c + 1))
                    else:
                        beta = (1 / (2 * (1 - rand))) ** (1 / (eta_c + 1))
                    
                    # Calculate children
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    # Apply bounds
                    lower, upper = self.gene_bounds[i]
                    c1 = max(min(c1, upper), lower)
                    c2 = max(min(c2, upper), lower)
                    
                    child1_genes.append(c1)
                    child2_genes.append(c2)
                else:
                    child1_genes.append(y1)
                    child2_genes.append(y2)
            else:
                child1_genes.append(parent1.genes[i])
                child2_genes.append(parent2.genes[i])
        
        child1 = Individual(genes=child1_genes, objectives=[0.0] * self.num_objectives)
        child2 = Individual(genes=child2_genes, objectives=[0.0] * self.num_objectives)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """Perform polynomial mutation."""
        eta_m = 20  # Distribution index for mutation
        
        mutated_genes = []
        for i, gene in enumerate(individual.genes):
            if random.random() <= self.mutation_prob:
                lower, upper = self.gene_bounds[i]
                
                delta1 = (gene - lower) / (upper - lower)
                delta2 = (upper - gene) / (upper - lower)
                
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                gene = gene + delta_q * (upper - lower)
                gene = max(min(gene, upper), lower)
            
            mutated_genes.append(gene)
        
        return Individual(genes=mutated_genes, objectives=[0.0] * self.num_objectives)
    
    def environmental_selection(
        self,
        population: List[Individual],
        offspring: List[Individual]
    ) -> List[Individual]:
        """Perform environmental selection using NSGA-II strategy."""
        
        # Combine population and offspring
        combined = population + offspring
        
        # Perform non-dominated sorting
        fronts = self.fast_non_dominated_sort(combined)
        
        # Select individuals for next generation
        next_population = []
        front_index = 0
        
        while len(next_population) + len(fronts[front_index]) <= self.population_size:
            # Calculate crowding distance for current front
            fronts[front_index] = self.calculate_crowding_distance(fronts[front_index])
            next_population.extend(fronts[front_index])
            front_index += 1
            
            if front_index >= len(fronts):
                break
        
        # Fill remaining slots from next front
        if len(next_population) < self.population_size and front_index < len(fronts):
            remaining_slots = self.population_size - len(next_population)
            last_front = self.calculate_crowding_distance(fronts[front_index])
            
            # Sort by crowding distance (descending)
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)
            next_population.extend(last_front[:remaining_slots])
        
        return next_population
    
    def get_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """Get the current Pareto front."""
        fronts = self.fast_non_dominated_sort(population)
        return fronts[0] if fronts else []
    
    def calculate_hypervolume(self, pareto_front: List[Individual]) -> float:
        """Calculate hypervolume indicator for the Pareto front."""
        if not pareto_front:
            return 0.0
        
        # Reference point (worst possible values)
        ref_point = [0.0] * self.num_objectives
        
        # Sort front by first objective
        sorted_front = sorted(pareto_front, key=lambda x: x.objectives[0])
        
        # Simple hypervolume calculation for 2D case
        if self.num_objectives == 2:
            hypervolume = 0.0
            prev_x = ref_point[0]
            
            for individual in sorted_front:
                width = individual.objectives[0] - prev_x
                height = individual.objectives[1] - ref_point[1]
                hypervolume += width * height
                prev_x = individual.objectives[0]
            
            return hypervolume
        
        # For higher dimensions, use approximation
        total_volume = 1.0
        for individual in sorted_front:
            volume = 1.0
            for obj_val in individual.objectives:
                volume *= max(0, obj_val - ref_point[0])
            total_volume += volume
        
        return total_volume / len(sorted_front)
    
    def evolve(
        self,
        objective_functions: List[Callable],
        code_analyzer,
        initial_population: Optional[List[Individual]] = None
    ) -> Tuple[List[Individual], Dict]:
        """Main evolution loop."""
        
        # Initialize population
        if initial_population:
            population = initial_population
        else:
            population = self.initialize_population()
        
        # Evaluate initial population
        population = self.evaluate_population(population, objective_functions, code_analyzer)
        
        best_hypervolume = 0.0
        stagnation_count = 0
        
        for generation in range(self.generations):
            logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Create offspring
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.extend([child1, child2])
            
            # Trim offspring to exact size
            offspring = offspring[:self.population_size]
            
            # Evaluate offspring
            offspring = self.evaluate_population(offspring, objective_functions, code_analyzer)
            
            # Environmental selection
            population = self.environmental_selection(population, offspring)
            
            # Calculate statistics
            pareto_front = self.get_pareto_front(population)
            current_hypervolume = self.calculate_hypervolume(pareto_front)
            
            # Track statistics
            gen_stats = {
                "generation": generation + 1,
                "pareto_front_size": len(pareto_front),
                "hypervolume": current_hypervolume,
                "avg_fitness": np.mean([ind.fitness for ind in population]),
                "best_fitness": max([ind.fitness for ind in population]),
                "objective_means": [
                    np.mean([ind.objectives[i] for ind in population])
                    for i in range(self.num_objectives)
                ]
            }
            
            self.generation_stats.append(gen_stats)
            self.pareto_front_history.append(copy.deepcopy(pareto_front))
            
            # Check for convergence
            if current_hypervolume > best_hypervolume:
                best_hypervolume = current_hypervolume
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"  Pareto front size: {len(pareto_front)}")
                logger.info(f"  Hypervolume: {current_hypervolume:.4f}")
                logger.info(f"  Average fitness: {gen_stats['avg_fitness']:.4f}")
            
            # Early stopping if stagnation
            if stagnation_count >= 20:
                logger.info(f"Early stopping at generation {generation + 1} due to stagnation")
                break
        
        # Final evaluation
        final_pareto_front = self.get_pareto_front(population)
        
        # Prepare results
        results = {
            "final_population": population,
            "pareto_front": final_pareto_front,
            "generation_stats": self.generation_stats,
            "pareto_front_history": self.pareto_front_history,
            "final_hypervolume": self.calculate_hypervolume(final_pareto_front),
            "total_generations": generation + 1
        }
        
        logger.info(f"Evolution completed after {generation + 1} generations")
        logger.info(f"Final Pareto front size: {len(final_pareto_front)}")
        
        return final_pareto_front, results
    
    def save_results(self, results: Dict, filename: str):
        """Save evolution results to file."""
        import json
        
        # Convert results to serializable format
        serializable_results = {
            "generation_stats": results["generation_stats"],
            "final_hypervolume": results["final_hypervolume"],
            "total_generations": results["total_generations"],
            "pareto_front_size": len(results["pareto_front"]),
            "pareto_front_objectives": [
                individual.objectives for individual in results["pareto_front"]
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def plot_pareto_front(self, pareto_front: List[Individual], save_path: str = None):
        """Plot the Pareto front (for 2D objectives)."""
        try:
            import matplotlib.pyplot as plt
            
            if self.num_objectives != 2:
                logger.warning("Plotting only supported for 2 objectives")
                return
            
            objectives_x = [ind.objectives[0] for ind in pareto_front]
            objectives_y = [ind.objectives[1] for ind in pareto_front]
            
            plt.figure(figsize=(10, 8))
            plt.scatter(objectives_x, objectives_y, c='red', s=50, alpha=0.7)
            plt.xlabel(f"Objective 1: {self.objectives[0]['name']}")
            plt.ylabel(f"Objective 2: {self.objectives[1]['name']}")
            plt.title("Pareto Front")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Pareto front plot saved to {save_path}")
            else:
                plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def plot_convergence(self, save_path: str = None):
        """Plot convergence statistics."""
        try:
            import matplotlib.pyplot as plt
            
            generations = [stat["generation"] for stat in self.generation_stats]
            hypervolumes = [stat["hypervolume"] for stat in self.generation_stats]
            avg_fitness = [stat["avg_fitness"] for stat in self.generation_stats]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Hypervolume plot
            ax1.plot(generations, hypervolumes, 'b-', linewidth=2)
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Hypervolume")
            ax1.set_title("Hypervolume Convergence")
            ax1.grid(True, alpha=0.3)
            
            # Average fitness plot
            ax2.plot(generations, avg_fitness, 'r-', linewidth=2)
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Average Fitness")
            ax2.set_title("Average Fitness Convergence")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Convergence plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")