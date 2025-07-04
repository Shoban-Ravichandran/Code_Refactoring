# src/refactoring/refactoring_engine.py
import os
import yaml
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..models.ensemble_model import EnsembleRefactoringModel
from ..genetic_algorithm.nsga2 import NSGA2, Individual
from ..genetic_algorithm.fitness_functions import create_objective_functions, CodeAnalyzer
from ..data.synthetic_generator import SyntheticDataGenerator
from ..utils.evaluation import RefactoringEvaluator
from ..utils.code_utils import CodeProcessor

logger = logging.getLogger(__name__)

class RefactoringEngine:
    """Main engine for AI-powered code refactoring."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the refactoring engine."""
        self.config = self._load_config(config_path)
        self.ensemble_model = None
        self.genetic_algorithm = None
        self.evaluator = None
        self.code_processor = CodeProcessor()
        
        # Initialize components
        self._setup_logging()
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "model": {
                "ensemble": {
                    "codebert_weight": 0.4,
                    "starcoder_weight": 0.6,
                    "fusion_method": "weighted_average"
                }
            },
            "genetic_algorithm": {
                "nsga2": {
                    "population_size": 50,
                    "generations": 25,
                    "crossover_probability": 0.8,
                    "mutation_probability": 0.2,
                    "tournament_size": 3,
                    "elite_size": 5
                },
                "objectives": [
                    {"name": "code_quality", "weight": 0.3, "maximize": True},
                    {"name": "readability", "weight": 0.3, "maximize": True},
                    {"name": "performance", "weight": 0.2, "maximize": True},
                    {"name": "maintainability", "weight": 0.2, "maximize": True}
                ]
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/refactoring_engine.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_components(self):
        """Initialize all components."""
        try:
            # Initialize ensemble model
            self.ensemble_model = EnsembleRefactoringModel(self.config)
            
            # Initialize genetic algorithm
            self.genetic_algorithm = NSGA2(self.config)
            
            # Initialize evaluator
            self.evaluator = RefactoringEvaluator(self.config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def load_pretrained_models(self, model_directory: str):
        """Load pre-trained models."""
        try:
            self.ensemble_model = EnsembleRefactoringModel.load_ensemble(
                model_directory, self.config
            )
            logger.info(f"Pre-trained models loaded from {model_directory}")
        except Exception as e:
            logger.error(f"Error loading pre-trained models: {e}")
            raise
    
    def refactor_code(
        self,
        source_code: str,
        optimization_target: str = "balanced",
        use_genetic_optimization: bool = True,
        max_generations: int = None
    ) -> Dict:
        """
        Refactor code using the AI-powered framework.
        
        Args:
            source_code: Original source code to refactor
            optimization_target: Target for optimization ('quality', 'readability', 'performance', 'maintainability', 'balanced')
            use_genetic_optimization: Whether to use genetic algorithm for optimization
            max_generations: Maximum generations for genetic algorithm (if None, uses config)
        
        Returns:
            Dictionary containing refactoring results
        """
        logger.info("Starting code refactoring process")
        
        try:
            # Step 1: Analyze original code
            code_analyzer = CodeAnalyzer(source_code)
            refactoring_candidates = code_analyzer.get_refactoring_candidates()
            
            logger.info(f"Found {len(refactoring_candidates)} refactoring candidates")
            
            # Step 2: Get initial refactoring suggestions from ensemble model
            initial_suggestions = self._get_initial_suggestions(
                source_code, refactoring_candidates
            )
            
            # Step 3: Optimize using genetic algorithm (if enabled)
            if use_genetic_optimization and self.genetic_algorithm:
                optimized_result = self._optimize_with_genetic_algorithm(
                    source_code, code_analyzer, optimization_target, max_generations
                )
            else:
                optimized_result = self._apply_best_suggestion(
                    source_code, initial_suggestions
                )
            
            # Step 4: Evaluate results
            evaluation_results = self.evaluator.evaluate_refactoring(
                source_code, optimized_result["refactored_code"]
            )
            
            # Step 5: Prepare final results
            final_results = {
                "original_code": source_code,
                "refactored_code": optimized_result["refactored_code"],
                "refactoring_pattern": optimized_result.get("pattern_type", "mixed"),
                "optimization_target": optimization_target,
                "candidates_found": refactoring_candidates,
                "initial_suggestions": initial_suggestions,
                "genetic_optimization_used": use_genetic_optimization,
                "evaluation": evaluation_results,
                "improvement_metrics": self._calculate_improvement_metrics(
                    evaluation_results
                ),
                "confidence_score": optimized_result.get("confidence", 0.5),
                "optimization_details": optimized_result.get("optimization_details", {})
            }
            
            logger.info("Code refactoring completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error during code refactoring: {e}")
            raise
    
    def _get_initial_suggestions(
        self, 
        source_code: str, 
        candidates: List[Dict]
    ) -> List[Dict]:
        """Get initial refactoring suggestions from the ensemble model."""
        suggestions = []
        
        try:
            # Get general refactoring suggestion
            general_result = self.ensemble_model.generate_refactored_code(
                source_code,
                use_voting=True,
                confidence_threshold=0.6
            )
            
            suggestions.append({
                "type": "general",
                "refactored_code": general_result["refactored_code"],
                "confidence": general_result["ensemble_confidence"],
                "selected_model": general_result["selected_model"],
                "quality_scores": general_result["quality_scores"]
            })
            
            # Get specific suggestions for each candidate
            for candidate in candidates[:3]:  # Limit to top 3 candidates
                pattern_type = candidate["type"]
                
                specific_result = self.ensemble_model.generate_refactored_code(
                    source_code,
                    use_voting=False,
                    confidence_threshold=0.5
                )
                
                suggestions.append({
                    "type": pattern_type,
                    "refactored_code": specific_result["refactored_code"],
                    "confidence": specific_result["ensemble_confidence"],
                    "candidate_info": candidate,
                    "quality_scores": specific_result["quality_scores"]
                })
            
        except Exception as e:
            logger.warning(f"Error getting initial suggestions: {e}")
            # Fallback to simple suggestion
            suggestions.append({
                "type": "fallback",
                "refactored_code": source_code,
                "confidence": 0.1,
                "quality_scores": {}
            })
        
        return suggestions
    
    def _optimize_with_genetic_algorithm(
        self,
        source_code: str,
        code_analyzer: CodeAnalyzer,
        optimization_target: str,
        max_generations: Optional[int]
    ) -> Dict:
        """Optimize refactoring using genetic algorithm."""
        logger.info("Starting genetic algorithm optimization")
        
        try:
            # Adjust objectives based on optimization target
            self._adjust_objectives_for_target(optimization_target)
            
            # Override generations if specified
            if max_generations:
                self.genetic_algorithm.generations = max_generations
            
            # Create objective functions
            objective_functions = create_objective_functions(self.config)
            
            # Run genetic algorithm
            pareto_front, evolution_results = self.genetic_algorithm.evolve(
                objective_functions=objective_functions,
                code_analyzer=code_analyzer
            )
            
            # Select best solution from Pareto front
            best_solution = self._select_best_from_pareto_front(
                pareto_front, optimization_target
            )
            
            return {
                "refactored_code": best_solution.refactored_code,
                "pattern_type": best_solution.pattern_type,
                "confidence": best_solution.fitness,
                "optimization_details": {
                    "pareto_front_size": len(pareto_front),
                    "generations_completed": evolution_results["total_generations"],
                    "final_hypervolume": evolution_results["final_hypervolume"],
                    "objectives": best_solution.objectives,
                    "genes": best_solution.genes
                }
            }
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            # Fallback to ensemble model result
            fallback_result = self.ensemble_model.generate_refactored_code(source_code)
            return {
                "refactored_code": fallback_result["refactored_code"],
                "pattern_type": "fallback",
                "confidence": fallback_result["ensemble_confidence"],
                "optimization_details": {"error": str(e)}
            }
    
    def _adjust_objectives_for_target(self, optimization_target: str):
        """Adjust objective weights based on optimization target."""
        if optimization_target == "quality":
            weights = [0.5, 0.2, 0.15, 0.15]  # Focus on code quality
        elif optimization_target == "readability":
            weights = [0.2, 0.5, 0.15, 0.15]  # Focus on readability
        elif optimization_target == "performance":
            weights = [0.15, 0.15, 0.5, 0.2]  # Focus on performance
        elif optimization_target == "maintainability":
            weights = [0.2, 0.15, 0.15, 0.5]  # Focus on maintainability
        else:  # balanced
            weights = [0.3, 0.3, 0.2, 0.2]  # Balanced approach
        
        # Update objective weights
        for i, objective in enumerate(self.config["genetic_algorithm"]["objectives"]):
            objective["weight"] = weights[i]
    
    def _select_best_from_pareto_front(
        self, 
        pareto_front: List[Individual], 
        optimization_target: str
    ) -> Individual:
        """Select the best solution from Pareto front based on target."""
        if not pareto_front:
            # Create dummy individual
            return Individual(
                genes=[0.5] * 8,
                objectives=[0.5] * 4,
                refactored_code="# No optimization result available"
            )
        
        if optimization_target == "balanced":
            # Select solution with highest overall fitness
            return max(pareto_front, key=lambda x: x.fitness)
        else:
            # Select based on specific objective
            objective_map = {
                "quality": 0,
                "readability": 1,
                "performance": 2,
                "maintainability": 3
            }
            
            objective_index = objective_map.get(optimization_target, 0)
            return max(pareto_front, key=lambda x: x.objectives[objective_index])
    
    def _apply_best_suggestion(
        self, 
        source_code: str, 
        suggestions: List[Dict]
    ) -> Dict:
        """Apply the best suggestion from initial suggestions."""
        if not suggestions:
            return {
                "refactored_code": source_code,
                "pattern_type": "none",
                "confidence": 0.0
            }
        
        # Select suggestion with highest confidence
        best_suggestion = max(suggestions, key=lambda x: x.get("confidence", 0))
        
        return {
            "refactored_code": best_suggestion["refactored_code"],
            "pattern_type": best_suggestion["type"],
            "confidence": best_suggestion["confidence"]
        }
    
    def _calculate_improvement_metrics(self, evaluation_results: Dict) -> Dict:
        """Calculate improvement metrics from evaluation results."""
        improvements = {}
        
        for metric, values in evaluation_results.items():
            if isinstance(values, dict) and "original" in values and "refactored" in values:
                original = values["original"]
                refactored = values["refactored"]
                
                if original > 0:
                    improvement = (refactored - original) / original * 100
                    improvements[f"{metric}_improvement_percent"] = improvement
                
                improvements[f"{metric}_absolute_change"] = refactored - original
        
        return improvements
    
    def batch_refactor(
        self,
        code_files: List[str],
        output_directory: str = "output/refactored",
        optimization_target: str = "balanced",
        use_genetic_optimization: bool = True
    ) -> Dict:
        """Refactor multiple code files in batch."""
        logger.info(f"Starting batch refactoring of {len(code_files)} files")
        
        os.makedirs(output_directory, exist_ok=True)
        
        results = {
            "processed_files": [],
            "failed_files": [],
            "summary_statistics": {},
            "total_files": len(code_files)
        }
        
        for i, file_path in enumerate(code_files):
            logger.info(f"Processing file {i+1}/{len(code_files)}: {file_path}")
            
            try:
                # Read source code
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Refactor code
                refactoring_result = self.refactor_code(
                    source_code,
                    optimization_target=optimization_target,
                    use_genetic_optimization=use_genetic_optimization
                )
                
                # Save refactored code
                output_file = os.path.join(
                    output_directory,
                    f"refactored_{os.path.basename(file_path)}"
                )
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(refactoring_result["refactored_code"])
                
                # Save detailed results
                result_file = output_file.replace('.py', '_results.json')
                import json
                with open(result_file, 'w') as f:
                    # Remove non-serializable items
                    serializable_result = {
                        k: v for k, v in refactoring_result.items() 
                        if k not in ['original_code', 'refactored_code']
                    }
                    json.dump(serializable_result, f, indent=2)
                
                results["processed_files"].append({
                    "input_file": file_path,
                    "output_file": output_file,
                    "result_file": result_file,
                    "confidence": refactoring_result["confidence_score"],
                    "pattern": refactoring_result["refactoring_pattern"],
                    "improvement_metrics": refactoring_result["improvement_metrics"]
                })
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                results["failed_files"].append({
                    "file": file_path,
                    "error": str(e)
                })
        
        # Calculate summary statistics
        if results["processed_files"]:
            confidences = [r["confidence"] for r in results["processed_files"]]
            results["summary_statistics"] = {
                "success_rate": len(results["processed_files"]) / len(code_files),
                "average_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences),
                "most_common_pattern": self._get_most_common_pattern(results["processed_files"])
            }
        
        logger.info(f"Batch refactoring completed. Processed: {len(results['processed_files'])}, Failed: {len(results['failed_files'])}")
        
        return results
    
    def _get_most_common_pattern(self, processed_files: List[Dict]) -> str:
        """Get the most commonly applied refactoring pattern."""
        from collections import Counter
        patterns = [file_result["pattern"] for file_result in processed_files]
        counter = Counter(patterns)
        return counter.most_common(1)[0][0] if counter else "none"
    
    def generate_training_data(
        self,
        output_directory: str = "data/training",
        dataset_size: int = 1000
    ) -> str:
        """Generate synthetic training data for model fine-tuning."""
        logger.info(f"Generating synthetic training data with {dataset_size} samples")
        
        try:
            generator = SyntheticDataGenerator(self.config)
            patterns = generator.generate_dataset(dataset_size)
            generator.save_dataset(patterns, output_directory)
            
            logger.info(f"Training data generated and saved to {output_directory}")
            return output_directory
            
        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            raise
    
    def evaluate_model_performance(
        self,
        test_dataset_path: str,
        metrics: List[str] = None
    ) -> Dict:
        """Evaluate model performance on test dataset."""
        logger.info(f"Evaluating model performance on {test_dataset_path}")
        
        try:
            # Load test dataset
            import json
            with open(test_dataset_path, 'r') as f:
                test_data = json.load(f)
            
            # Evaluate ensemble model
            performance_results = self.ensemble_model.evaluate_ensemble_performance(
                test_data, metrics
            )
            
            logger.info("Model evaluation completed")
            return performance_results
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            raise


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Powered Code Refactoring Framework")
    parser.add_argument("--input", "-i", required=True, help="Input code file")
    parser.add_argument("--output", "-o", help="Output file for refactored code")
    parser.add_argument("--config", "-c", default="config/model_config.yaml", help="Configuration file")
    parser.add_argument("--target", "-t", default="balanced", 
                       choices=["balanced", "quality", "readability", "performance", "maintainability"],
                       help="Optimization target")
    parser.add_argument("--no-genetic", action="store_true", help="Disable genetic algorithm optimization")
    parser.add_argument("--generations", "-g", type=int, help="Maximum generations for genetic algorithm")
    parser.add_argument("--batch", "-b", help="Directory containing multiple Python files to refactor")
    parser.add_argument("--model-dir", "-m", help="Directory containing pre-trained models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize refactoring engine
        engine = RefactoringEngine(args.config)
        
        # Load pre-trained models if specified
        if args.model_dir:
            engine.load_pretrained_models(args.model_dir)
        
        if args.batch:
            # Batch processing
            import glob
            code_files = glob.glob(os.path.join(args.batch, "*.py"))
            
            if not code_files:
                print(f"No Python files found in {args.batch}")
                return
            
            output_dir = args.output or "output/batch_refactored"
            results = engine.batch_refactor(
                code_files=code_files,
                output_directory=output_dir,
                optimization_target=args.target,
                use_genetic_optimization=not args.no_genetic
            )
            
            print(f"\nBatch refactoring completed!")
            print(f"Processed: {len(results['processed_files'])} files")
            print(f"Failed: {len(results['failed_files'])} files")
            print(f"Success rate: {results['summary_statistics'].get('success_rate', 0):.2%}")
            print(f"Average confidence: {results['summary_statistics'].get('average_confidence', 0):.3f}")
            
        else:
            # Single file processing
            if not os.path.exists(args.input):
                print(f"Input file {args.input} not found")
                return
            
            # Read input code
            with open(args.input, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Refactor code
            result = engine.refactor_code(
                source_code=source_code,
                optimization_target=args.target,
                use_genetic_optimization=not args.no_genetic,
                max_generations=args.generations
            )
            
            # Save output
            output_file = args.output or args.input.replace('.py', '_refactored.py')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result["refactored_code"])
            
            # Save detailed results
            result_file = output_file.replace('.py', '_results.json')
            import json
            with open(result_file, 'w') as f:
                serializable_result = {
                    k: v for k, v in result.items() 
                    if k not in ['original_code', 'refactored_code']
                }
                json.dump(serializable_result, f, indent=2)
            
            # Print summary
            print(f"\nRefactoring completed!")
            print(f"Input: {args.input}")
            print(f"Output: {output_file}")
            print(f"Results: {result_file}")
            print(f"Pattern applied: {result['refactoring_pattern']}")
            print(f"Confidence: {result['confidence_score']:.3f}")
            print(f"Optimization target: {result['optimization_target']}")
            
            if result['improvement_metrics']:
                print("\nImprovements:")
                for metric, value in result['improvement_metrics'].items():
                    if 'percent' in metric:
                        print(f"  {metric}: {value:.1f}%")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())