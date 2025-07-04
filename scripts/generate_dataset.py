"""
Script to generate synthetic refactoring datasets for training.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.synthetic_generator import SyntheticDataGenerator
from data.code_patterns import CodePatternGenerator

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/dataset_generation.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def generate_synthetic_dataset(args):
    """Generate synthetic refactoring dataset."""
    logger.info(f"Generating synthetic dataset with {args.size} samples")
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    
    # Initialize generator
    generator = SyntheticDataGenerator(config)
    
    # Generate patterns
    patterns = generator.generate_dataset(args.size)
    
    # Save dataset
    output_dir = args.output or "data/synthetic_dataset"
    generator.save_dataset(patterns, output_dir)
    
    logger.info(f"Dataset generated successfully: {len(patterns)} patterns saved to {output_dir}")
    
    # Generate statistics
    generate_dataset_statistics(patterns, output_dir)

def generate_code_patterns(args):
    """Generate specific code patterns for training."""
    logger.info("Generating specific code patterns")
    
    config = load_config(args.config) if args.config else {}
    pattern_generator = CodePatternGenerator(config)
    
    patterns = pattern_generator.generate_all_patterns(args.samples_per_pattern)
    
    output_dir = args.output or "data/code_patterns"
    pattern_generator.save_patterns(patterns, output_dir)
    
    logger.info(f"Code patterns generated: {len(patterns)} patterns saved to {output_dir}")

def generate_dataset_statistics(patterns, output_dir):
    """Generate and save dataset statistics."""
    from collections import Counter
    
    stats = {
        "total_patterns": len(patterns),
        "pattern_distribution": Counter(p.pattern_type for p in patterns),
        "complexity_levels": {
            "low": sum(1 for p in patterns if p.complexity_reduction < 0.3),
            "medium": sum(1 for p in patterns if 0.3 <= p.complexity_reduction < 0.7),
            "high": sum(1 for p in patterns if p.complexity_reduction >= 0.7)
        },
        "average_improvements": {
            "readability": sum(p.quality_improvement.get("readability", 0) for p in patterns) / len(patterns),
            "maintainability": sum(p.quality_improvement.get("maintainability", 0) for p in patterns) / len(patterns),
            "performance": sum(p.quality_improvement.get("performance", 0) for p in patterns) / len(patterns)
        }
    }
    
    # Save statistics
    import json
    with open(os.path.join(output_dir, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Print summary
    print("\nDataset Statistics:")
    print(f"Total patterns: {stats['total_patterns']}")
    print("\nPattern distribution:")
    for pattern, count in stats["pattern_distribution"].items():
        print(f"  {pattern}: {count}")
    print("\nComplexity levels:")
    for level, count in stats["complexity_levels"].items():
        print(f"  {level}: {count}")

def augment_existing_dataset(args):
    """Augment existing dataset with variations."""
    logger.info(f"Augmenting dataset from {args.input}")
    
    import json
    
    # Load existing dataset
    with open(args.input, 'r') as f:
        existing_data = json.load(f)
    
    config = load_config(args.config) if args.config else {}
    generator = SyntheticDataGenerator(config)
    
    # Create augmented patterns
    augmented_patterns = []
    for item in existing_data:
        # Convert to CodePattern object
        pattern = generator.create_pattern_from_dict(item)
        
        # Generate variations
        variations = generator.generate_pattern_variations(pattern, args.augmentation_factor)
        augmented_patterns.extend(variations)
    
    # Save augmented dataset
    output_dir = args.output or "data/augmented_dataset"
    generator.save_dataset(augmented_patterns, output_dir)
    
    logger.info(f"Dataset augmented: {len(augmented_patterns)} patterns saved to {output_dir}")

def validate_dataset(args):
    """Validate generated dataset."""
    logger.info(f"Validating dataset at {args.dataset}")
    
    import json
    import ast
    
    validation_results = {
        "total_samples": 0,
        "valid_samples": 0,
        "syntax_errors": [],
        "pattern_coverage": {},
        "quality_issues": []
    }
    
    # Load dataset
    with open(os.path.join(args.dataset, "refactoring_patterns.json"), 'r') as f:
        data = json.load(f)
    
    validation_results["total_samples"] = len(data)
    
    for i, sample in enumerate(data):
        try:
            # Validate syntax
            ast.parse(sample["original_code"])
            ast.parse(sample["refactored_code"])
            
            # Check pattern type
            pattern_type = sample["pattern_type"]
            validation_results["pattern_coverage"][pattern_type] = \
                validation_results["pattern_coverage"].get(pattern_type, 0) + 1
            
            # Check quality metrics
            quality = sample.get("quality_improvement", {})
            if not quality or all(v == 0 for v in quality.values()):
                validation_results["quality_issues"].append(f"Sample {i}: No quality improvements")
            
            validation_results["valid_samples"] += 1
            
        except SyntaxError as e:
            validation_results["syntax_errors"].append(f"Sample {i}: {str(e)}")
        except Exception as e:
            validation_results["syntax_errors"].append(f"Sample {i}: Unexpected error - {str(e)}")
    
    # Save validation results
    with open(os.path.join(args.dataset, "validation_results.json"), 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Print summary
    print(f"\nValidation Results:")
    print(f"Total samples: {validation_results['total_samples']}")
    print(f"Valid samples: {validation_results['valid_samples']}")
    print(f"Success rate: {validation_results['valid_samples']/validation_results['total_samples']:.2%}")
    print(f"Syntax errors: {len(validation_results['syntax_errors'])}")
    print(f"Quality issues: {len(validation_results['quality_issues'])}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic refactoring datasets")
    
    # Global arguments
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Synthetic dataset generation
    synthetic_parser = subparsers.add_parser("synthetic", help="Generate synthetic dataset")
    synthetic_parser.add_argument("--size", "-s", type=int, default=1000, help="Dataset size")
    synthetic_parser.add_argument("--output", "-o", help="Output directory")
    
    # Code patterns generation
    patterns_parser = subparsers.add_parser("patterns", help="Generate code patterns")
    patterns_parser.add_argument("--samples-per-pattern", "-n", type=int, default=100, 
                                help="Samples per pattern type")
    patterns_parser.add_argument("--output", "-o", help="Output directory")
    
    # Dataset augmentation
    augment_parser = subparsers.add_parser("augment", help="Augment existing dataset")
    augment_parser.add_argument("--input", "-i", required=True, help="Input dataset file")
    augment_parser.add_argument("--output", "-o", help="Output directory")
    augment_parser.add_argument("--augmentation-factor", "-f", type=int, default=3,
                               help="Augmentation factor")
    
    # Dataset validation
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--dataset", "-d", required=True, help="Dataset directory")
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "synthetic":
            generate_synthetic_dataset(args)
        elif args.command == "patterns":
            generate_code_patterns(args)
        elif args.command == "augment":
            augment_existing_dataset(args)
        elif args.command == "validate":
            validate_dataset(args)
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error in {args.command}: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())