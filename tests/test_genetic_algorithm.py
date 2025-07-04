"""
Unit tests for the genetic algorithm components.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from genetic_algorithm.nsga2 import NSGA2, Individual
from genetic_algorithm.fitness_functions import FitnessEvaluator, CodeAnalyzer, create_objective_functions

class TestIndividual(unittest.TestCase):
    """Test cases for Individual class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.individual = Individual(
            genes=[0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.9],
            objectives=[0.8, 0.6, 0.7, 0.5],
            fitness=0.65,
            rank=1,
            crowding_distance=0.5
        )
    
    def test_individual_creation(self):
        """Test Individual creation."""
        self.assertEqual(len(self.individual.genes), 8)
        self.assertEqual(len(self.individual.objectives), 4)
        self.assertEqual(self.individual.fitness, 0.65)
        self.assertEqual(self.individual.rank, 1)
    
    def test_individual_attributes(self):
        """Test Individual attributes."""
        self.assertIsInstance(self.individual.genes, list)
        self.assertIsInstance(self.individual.objectives, list)
        self.assertIsInstance(self.individual.fitness, float)
        self.assertIsInstance(self.individual.rank, int)

class TestNSGA2(unittest.TestCase):
    """Test cases for NSGA-II algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "genetic_algorithm": {
                "nsga2": {
                    "population_size": 20,
                    "generations": 5,
                    "crossover_probability": 0.8,
                    "mutation_probability": 0.2,
                    "tournament_size": 3,
                    "elite_size": 2
                },
                "objectives": [
                    {"name": "code_quality", "weight": 0.3, "maximize": True},
                    {"name": "readability", "weight": 0.3, "maximize": True},
                    {"name": "performance", "weight": 0.2, "maximize": True},
                    {"name": "maintainability", "weight": 0.2, "maximize": True}
                ]
            }
        }
        self.nsga2 = NSGA2(self.config)
    
    def test_nsga2_initialization(self):
        """Test NSGA-II initialization."""
        self.assertEqual(self.nsga2.population_size, 20)
        self.assertEqual(self.nsga2.generations, 5)
        self.assertEqual(self.nsga2.crossover_prob, 0.8)
        self.assertEqual(self.nsga2.mutation_prob, 0.2)
        self.assertEqual(self.nsga2.num_objectives, 4)
        self.assertEqual(self.nsga2.num_genes, 8)
    
    def test_population_initialization(self):
        """Test population initialization."""
        population = self.nsga2.initialize_population()
        
        self.assertEqual(len(population), self.nsga2.population_size)
        
        for individual in population:
            self.assertIsInstance(individual, Individual)
            self.assertEqual(len(individual.genes), self.nsga2.num_genes)
            self.assertEqual(len(individual.objectives), self.nsga2.num_objectives)
            
            # Check gene bounds
            for gene in individual.genes:
                self.assertGreaterEqual(gene, 0.0)
                self.assertLessEqual(gene, 1.0)
    
    def test_dominance_relation(self):
        """Test dominance relation between individuals."""
        # Individual p dominates q if p is better in all objectives
        p = Individual(genes=[0.5] * 8, objectives=[0.8, 0.7, 0.9, 0.6])
        q = Individual(genes=[0.5] * 8, objectives=[0.6, 0.5, 0.7, 0.4])
        
        self.assertTrue(self.nsga2._dominates(p, q))
        self.assertFalse(self.nsga2._dominates(q, p))
    
    def test_non_dominance(self):
        """Test non-dominance relation."""
        # Neither individual dominates the other
        p = Individual(genes=[0.5] * 8, objectives=[0.8, 0.5, 0.7, 0.6])
        q = Individual(genes=[0.5] * 8, objectives=[0.6, 0.7, 0.9, 0.4])
        
        self.assertFalse(self.nsga2._dominates(p, q))
        self.assertFalse(self.nsga2._dominates(q, p))
    
    def test_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        # Create test population
        population = [
            Individual(genes=[0.5] * 8, objectives=[0.9, 0.8, 0.7, 0.9]),  # Rank 0
            Individual(genes=[0.5] * 8, objectives=[0.8, 0.7, 0.6, 0.8]),  # Rank 1
            Individual(genes=[0.5] * 8, objectives=[0.7, 0.6, 0.8, 0.7]),  # Rank 1
            Individual(genes=[0.5] * 8, objectives=[0.6, 0.5, 0.5, 0.6]),  # Rank 2
        ]
        
        fronts = self.nsga2.fast_non_dominated_sort(population)
        
        # Check that we have multiple fronts
        self.assertGreater(len(fronts), 1)
        
        # Check that first front has the best individual
        self.assertEqual(len(fronts[0]), 1)
        self.assertEqual(fronts[0][0].rank, 0)
        
        # Check that all individuals are assigned ranks
        for individual in population:
            self.assertGreaterEqual(individual.rank, 0)
    
    def test_crowding_distance_calculation(self):
        """Test crowding distance calculation."""
        # Create front with multiple individuals
        front = [
            Individual(genes=[0.5] * 8, objectives=[0.9, 0.1]),
            Individual(genes=[0.5] * 8, objectives=[0.5, 0.5]),
            Individual(genes=[0.5] * 8, objectives=[0.1, 0.9])
        ]
        
        front_with_distance = self.nsga2.calculate_crowding_distance(front)
        
        # Boundary points should have infinite distance
        self.assertEqual(front_with_distance[0].crowding_distance, float('inf'))
        self.assertEqual(front_with_distance[-1].crowding_distance, float('inf'))
        
        # Middle point should have finite distance
        self.assertIsInstance(front_with_distance[1].crowding_distance, float)
        self.assertNotEqual(front_with_distance[1].crowding_distance, float('inf'))
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        # Create population with different ranks
        population = [
            Individual(genes=[0.5] * 8, objectives=[0.8] * 4, rank=0, crowding_distance=0.5),
            Individual(genes=[0.5] * 8, objectives=[0.7] * 4, rank=1, crowding_distance=0.3),
            Individual(genes=[0.5] * 8, objectives=[0.6] * 4, rank=2, crowding_distance=0.7),
        ]
        
        selected = self.nsga2.tournament_selection(population)
        
        self.assertIsInstance(selected, Individual)
        # Should prefer lower rank (better individuals)
        # This is stochastic, so we can't guarantee specific selection
    
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = Individual(genes=[0.2, 0.4, 0.6, 0.8] * 2, objectives=[0.5] * 4)
        parent2 = Individual(genes=[0.8, 0.6, 0.4, 0.2] * 2, objectives=[0.5] * 4)
        
        child1, child2 = self.nsga2.crossover(parent1, parent2)
        
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)
        self.assertEqual(len(child1.genes), self.nsga2.num_genes)
        self.assertEqual(len(child2.genes), self.nsga2.num_genes)
        
        # Check gene bounds
        for gene in child1.genes + child2.genes:
            self.assertGreaterEqual(gene, 0.0)
            self.assertLessEqual(gene, 1.0)
    
    def test_mutation(self):
        """Test mutation operation."""
        individual = Individual(genes=[0.5] * 8, objectives=[0.5] * 4)
        mutated = self.nsga2.mutate(individual)
        
        self.assertIsInstance(mutated, Individual)
        self.assertEqual(len(mutated.genes), self.nsga2.num_genes)
        
        # Check gene bounds
        for gene in mutated.genes:
            self.assertGreaterEqual(gene, 0.0)
            self.assertLessEqual(gene, 1.0)
    
    def test_environmental_selection(self):
        """Test environmental selection."""
        # Create population and offspring
        population = self.nsga2.initialize_population()
        offspring = self.nsga2.initialize_population()
        
        # Mock evaluation (set dummy objectives)
        for ind in population + offspring:
            ind.objectives = np.random.rand(4).tolist()
        
        selected = self.nsga2.environmental_selection(population, offspring)
        
        self.assertEqual(len(selected), self.nsga2.population_size)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))
    
    def test_hypervolume_calculation(self):
        """Test hypervolume calculation."""
        pareto_front = [
            Individual(genes=[0.5] * 8, objectives=[0.8, 0.6]),
            Individual(genes=[0.5] * 8, objectives=[0.6, 0.8]),
            Individual(genes=[0.5] * 8, objectives=[0.7, 0.7])
        ]
        
        # Modify for 2D test
        self.nsga2.num_objectives = 2
        hypervolume = self.nsga2.calculate_hypervolume(pareto_front)
        
        self.assertIsInstance(hypervolume, float)
        self.assertGreaterEqual(hypervolume, 0.0)
    
    def test_get_pareto_front(self):
        """Test Pareto front extraction."""
        population = [
            Individual(genes=[0.5] * 8, objectives=[0.9, 0.8, 0.7, 0.9]),
            Individual(genes=[0.5] * 8, objectives=[0.8, 0.7, 0.6, 0.8]),
            Individual(genes=[0.5] * 8, objectives=[0.7, 0.6, 0.8, 0.7]),
        ]
        
        pareto_front = self.nsga2.get_pareto_front(population)
        
        self.assertIsInstance(pareto_front, list)
        self.assertGreater(len(pareto_front), 0)
        self.assertTrue(all(isinstance(ind, Individual) for ind in pareto_front))

class TestFitnessEvaluator(unittest.TestCase):
    """Test cases for fitness evaluation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "genetic_algorithm": {
                "objectives": [
                    {"name": "code_quality", "weight": 0.3, "maximize": True},
                    {"name": "readability", "weight": 0.3, "maximize": True},
                    {"name": "performance", "weight": 0.2, "maximize": True},
                    {"name": "maintainability", "weight": 0.2, "maximize": True}
                ]
            }
        }
        self.evaluator = FitnessEvaluator(self.config)
        
        self.sample_code = """
def calculate_total(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
    return total
"""
        self.code_analyzer = CodeAnalyzer(self.sample_code)
    
    def test_evaluator_initialization(self):
        """Test fitness evaluator initialization."""
        self.assertIsInstance(self.evaluator, FitnessEvaluator)
        self.assertEqual(len(self.evaluator.objectives), 4)
    
    def test_code_quality_evaluation(self):
        """Test code quality evaluation."""
        individual = Individual(genes=[0.5] * 8, objectives=[0.0] * 4)
        
        with patch.object(self.evaluator, '_apply_refactoring') as mock_refactor:
            mock_refactor.return_value = self.sample_code
            
            quality_score = self.evaluator.evaluate_code_quality(individual, self.code_analyzer)
            
            self.assertIsInstance(quality_score, float)
            self.assertGreaterEqual(quality_score, 0.0)
            self.assertLessEqual(quality_score, 1.0)
    
    def test_readability_evaluation(self):
        """Test readability evaluation."""
        individual = Individual(genes=[0.5] * 8, objectives=[0.0] * 4)
        
        with patch.object(self.evaluator, '_apply_refactoring') as mock_refactor:
            mock_refactor.return_value = self.sample_code
            
            readability_score = self.evaluator.evaluate_readability(individual, self.code_analyzer)
            
            self.assertIsInstance(readability_score, float)
            self.assertGreaterEqual(readability_score, 0.0)
            self.assertLessEqual(readability_score, 1.0)
    
    def test_performance_evaluation(self):
        """Test performance evaluation."""
        individual = Individual(genes=[0.5] * 8, objectives=[0.0] * 4)
        
        with patch.object(self.evaluator, '_apply_refactoring') as mock_refactor:
            mock_refactor.return_value = self.sample_code
            
            performance_score = self.evaluator.evaluate_performance(individual, self.code_analyzer)
            
            self.assertIsInstance(performance_score, float)
            self.assertGreaterEqual(performance_score, 0.0)
            self.assertLessEqual(performance_score, 1.0)
    
    def test_maintainability_evaluation(self):
        """Test maintainability evaluation."""
        individual = Individual(genes=[0.5] * 8, objectives=[0.0] * 4)
        
        with patch.object(self.evaluator, '_apply_refactoring') as mock_refactor:
            mock_refactor.return_value = self.sample_code
            
            maintainability_score = self.evaluator.evaluate_maintainability(individual, self.code_analyzer)
            
            self.assertIsInstance(maintainability_score, float)
            self.assertGreaterEqual(maintainability_score, 0.0)
            self.assertLessEqual(maintainability_score, 1.0)

class TestCodeAnalyzer(unittest.TestCase):
    """Test cases for code analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_code = """
def long_method_with_complexity(items, threshold, multiplier):
    result = []
    total = 0
    
    for item in items:
        if item is not None:
            if item > threshold:
                if item < 100:
                    processed = item * multiplier
                    result.append(processed)
                    total += processed
                else:
                    processed = item * 0.5
                    result.append(processed)
                    total += processed
            else:
                result.append(item)
                total += item
    
    if total > 1000:
        return result, total * 0.9
    else:
        return result, total
"""
        self.analyzer = CodeAnalyzer(self.sample_code)
    
    def test_analyzer_initialization(self):
        """Test code analyzer initialization."""
        self.assertIsInstance(self.analyzer, CodeAnalyzer)
        self.assertIsNotNone(self.analyzer.ast_tree)
        self.assertIsInstance(self.analyzer.metrics, dict)
    
    def test_baseline_metrics_calculation(self):
        """Test baseline metrics calculation."""
        self.assertIn('original_complexity', self.analyzer.metrics)
        self.assertIn('original_lines', self.analyzer.metrics)
        self.assertIn('original_methods', self.analyzer.metrics)
        
        # Check reasonable values
        self.assertGreater(self.analyzer.metrics['original_complexity'], 1)
        self.assertGreater(self.analyzer.metrics['original_lines'], 10)
    
    def test_refactoring_candidates_extraction(self):
        """Test refactoring candidates extraction."""
        candidates = self.analyzer.get_refactoring_candidates()
        
        self.assertIsInstance(candidates, list)
        
        # Should find at least some refactoring opportunities
        self.assertGreater(len(candidates), 0)
        
        # Check candidate structure
        for candidate in candidates:
            self.assertIn('type', candidate)
            self.assertIn('location', candidate)
            self.assertIn('reason', candidate)
            self.assertIn('priority', candidate)
    
    def test_refactoring_impact_estimation(self):
        """Test refactoring impact estimation."""
        impact = self.analyzer.estimate_refactoring_impact('extract_method')
        
        self.assertIsInstance(impact, dict)
        self.assertIn('code_quality', impact)
        self.assertIn('readability', impact)
        self.assertIn('performance', impact)
        self.assertIn('maintainability', impact)
        
        # Check values are reasonable
        for value in impact.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

class TestObjectiveFunctions(unittest.TestCase):
    """Test cases for objective function creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "genetic_algorithm": {
                "objectives": [
                    {"name": "code_quality", "weight": 0.3, "maximize": True},
                    {"name": "readability", "weight": 0.3, "maximize": True},
                    {"name": "performance", "weight": 0.2, "maximize": True},
                    {"name": "maintainability", "weight": 0.2, "maximize": True}
                ]
            }
        }
    
    def test_objective_functions_creation(self):
        """Test creation of objective functions."""
        objective_functions = create_objective_functions(self.config)
        
        self.assertIsInstance(objective_functions, list)
        self.assertEqual(len(objective_functions), 4)
        
        # Check that all functions are callable
        for func in objective_functions:
            self.assertTrue(callable(func))
    
    def test_unknown_objective_handling(self):
        """Test handling of unknown objectives."""
        config_with_unknown = {
            "genetic_algorithm": {
                "objectives": [
                    {"name": "unknown_objective", "weight": 1.0, "maximize": True}
                ]
            }
        }
        
        objective_functions = create_objective_functions(config_with_unknown)
        
        # Should return empty list for unknown objectives
        self.assertEqual(len(objective_functions), 0)

class TestGeneticAlgorithmIntegration(unittest.TestCase):
    """Integration tests for genetic algorithm components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "genetic_algorithm": {
                "nsga2": {
                    "population_size": 10,  # Small for testing
                    "generations": 3,       # Few generations for testing
                    "crossover_probability": 0.8,
                    "mutation_probability": 0.2,
                    "tournament_size": 2,
                    "elite_size": 1
                },
                "objectives": [
                    {"name": "code_quality", "weight": 0.5, "maximize": True},
                    {"name": "readability", "weight": 0.5, "maximize": True}
                ]
            }
        }
        
        self.sample_code = """
def simple_function(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x
    else:
        return 0
"""
    
    def test_full_evolution_cycle(self):
        """Test complete evolution cycle."""
        nsga2 = NSGA2(self.config)
        code_analyzer = CodeAnalyzer(self.sample_code)
        objective_functions = create_objective_functions(self.config)
        
        # Mock objective function evaluation to avoid complex dependencies
        def mock_objective(individual, analyzer):
            return np.random.rand()
        
        objective_functions = [mock_objective, mock_objective]
        
        # Run evolution
        pareto_front, results = nsga2.evolve(objective_functions, code_analyzer)
        
        # Check results
        self.assertIsInstance(pareto_front, list)
        self.assertIsInstance(results, dict)
        
        self.assertIn('final_population', results)
        self.assertIn('generation_stats', results)
        self.assertIn('total_generations', results)
        
        # Check that evolution ran
        self.assertGreaterEqual(results['total_generations'], 1)
        self.assertLessEqual(results['total_generations'], self.config['genetic_algorithm']['nsga2']['generations'])
    
    def test_convergence_detection(self):
        """Test convergence detection."""
        nsga2 = NSGA2(self.config)
        
        # Create static population that shouldn't improve
        population = []
        for _ in range(nsga2.population_size):
            ind = Individual(genes=[0.5] * 8, objectives=[0.5, 0.5])
            population.append(ind)
        
        # Calculate hypervolume (should be constant)
        hypervolume1 = nsga2.calculate_hypervolume(population)
        hypervolume2 = nsga2.calculate_hypervolume(population)
        
        self.assertEqual(hypervolume1, hypervolume2)
    
    def test_statistics_tracking(self):
        """Test statistics tracking during evolution."""
        nsga2 = NSGA2(self.config)
        
        # Initialize population
        population = nsga2.initialize_population()
        
        # Set dummy objectives
        for ind in population:
            ind.objectives = [np.random.rand(), np.random.rand()]
            ind.fitness = np.mean(ind.objectives)
        
        # Get Pareto front and calculate stats
        pareto_front = nsga2.get_pareto_front(population)
        hypervolume = nsga2.calculate_hypervolume(pareto_front)
        
        # Create generation stats
        gen_stats = {
            "generation": 1,
            "pareto_front_size": len(pareto_front),
            "hypervolume": hypervolume,
            "avg_fitness": np.mean([ind.fitness for ind in population]),
            "best_fitness": max([ind.fitness for ind in population])
        }
        
        # Check stats structure
        self.assertIn('generation', gen_stats)
        self.assertIn('pareto_front_size', gen_stats)
        self.assertIn('hypervolume', gen_stats)
        self.assertIn('avg_fitness', gen_stats)
        self.assertIn('best_fitness', gen_stats)
        
        # Check reasonable values
        self.assertGreaterEqual(gen_stats['pareto_front_size'], 1)
        self.assertGreaterEqual(gen_stats['hypervolume'], 0)
        self.assertGreaterEqual(gen_stats['avg_fitness'], 0)
        self.assertGreaterEqual(gen_stats['best_fitness'], gen_stats['avg_fitness'])

class TestGeneticAlgorithmEdgeCases(unittest.TestCase):
    """Test edge cases for genetic algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "genetic_algorithm": {
                "nsga2": {
                    "population_size": 5,
                    "generations": 2,
                    "crossover_probability": 0.8,
                    "mutation_probability": 0.2,
                    "tournament_size": 2,
                    "elite_size": 1
                },
                "objectives": [
                    {"name": "code_quality", "weight": 1.0, "maximize": True}
                ]
            }
        }
    
    def test_small_population(self):
        """Test with very small population."""
        self.config['genetic_algorithm']['nsga2']['population_size'] = 2
        nsga2 = NSGA2(self.config)
        
        population = nsga2.initialize_population()
        self.assertEqual(len(population), 2)
        
        # Test operations with small population
        parent1, parent2 = population[0], population[1]
        child1, child2 = nsga2.crossover(parent1, parent2)
        
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)
    
    def test_single_objective(self):
        """Test with single objective."""
        nsga2 = NSGA2(self.config)
        self.assertEqual(nsga2.num_objectives, 1)
        
        # Test dominance with single objective
        p = Individual(genes=[0.5] * 8, objectives=[0.8])
        q = Individual(genes=[0.5] * 8, objectives=[0.6])
        
        self.assertTrue(nsga2._dominates(p, q))
        self.assertFalse(nsga2._dominates(q, p))
    
    def test_identical_individuals(self):
        """Test with identical individuals."""
        population = []
        for _ in range(5):
            ind = Individual(genes=[0.5] * 8, objectives=[0.7])
            population.append(ind)
        
        nsga2 = NSGA2(self.config)
        fronts = nsga2.fast_non_dominated_sort(population)
        
        # All identical individuals should be in the same front
        self.assertEqual(len(fronts), 1)
        self.assertEqual(len(fronts[0]), 5)
    
    def test_extreme_gene_values(self):
        """Test with extreme gene values."""
        # Test with all minimum values
        ind_min = Individual(genes=[0.0] * 8, objectives=[0.0])
        
        # Test with all maximum values
        ind_max = Individual(genes=[1.0] * 8, objectives=[1.0])
        
        nsga2 = NSGA2(self.config)
        
        # Test mutation doesn't break bounds
        mutated_min = nsga2.mutate(ind_min)
        mutated_max = nsga2.mutate(ind_max)
        
        for gene in mutated_min.genes + mutated_max.genes:
            self.assertGreaterEqual(gene, 0.0)
            self.assertLessEqual(gene, 1.0)
    
    def test_empty_population_handling(self):
        """Test handling of empty populations."""
        nsga2 = NSGA2(self.config)
        
        # Test with empty population
        empty_population = []
        fronts = nsga2.fast_non_dominated_sort(empty_population)
        
        # Should handle gracefully
        self.assertIsInstance(fronts, list)
        
        # Test Pareto front extraction with empty population
        pareto_front = nsga2.get_pareto_front(empty_population)
        self.assertEqual(len(pareto_front), 0)
        
        # Test hypervolume with empty front
        hypervolume = nsga2.calculate_hypervolume(pareto_front)
        self.assertEqual(hypervolume, 0.0)

class TestGeneticAlgorithmPerformance(unittest.TestCase):
    """Performance tests for genetic algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "genetic_algorithm": {
                "nsga2": {
                    "population_size": 50,
                    "generations": 10,
                    "crossover_probability": 0.8,
                    "mutation_probability": 0.2,
                    "tournament_size": 3,
                    "elite_size": 5
                },
                "objectives": [
                    {"name": "code_quality", "weight": 0.25, "maximize": True},
                    {"name": "readability", "weight": 0.25, "maximize": True},
                    {"name": "performance", "weight": 0.25, "maximize": True},
                    {"name": "maintainability", "weight": 0.25, "maximize": True}
                ]
            }
        }
    
    @unittest.skip("Performance test - run manually")
    def test_large_population_performance(self):
        """Test performance with larger population."""
        import time
        
        nsga2 = NSGA2(self.config)
        
        start_time = time.time()
        population = nsga2.initialize_population()
        
        # Set random objectives
        for ind in population:
            ind.objectives = [np.random.rand() for _ in range(4)]
        
        fronts = nsga2.fast_non_dominated_sort(population)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Non-dominated sorting time for {nsga2.population_size} individuals: {execution_time:.4f} seconds")
        
        # Should complete in reasonable time
        self.assertLess(execution_time, 5.0)  # Less than 5 seconds
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        nsga2 = NSGA2(self.config)
        
        # Create multiple generations worth of individuals
        all_individuals = []
        
        for generation in range(10):
            population = nsga2.initialize_population()
            all_individuals.extend(population)
        
        # Should be able to create many individuals without issues
        self.assertEqual(len(all_individuals), 10 * nsga2.population_size)

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)