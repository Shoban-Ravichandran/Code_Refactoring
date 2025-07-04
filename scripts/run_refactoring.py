"""
Script to run code refactoring on individual files or projects.
"""

import os
import sys
import argparse
import logging
import json
import glob
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from refactoring.refactoring_engine import RefactoringEngine
from utils.evaluation import RefactoringEvaluator

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/refactoring.log'),
            logging.StreamHandler()
        ]
    )

def refactor_single_file(args):
    """Refactor a single file."""
    logger.info(f"Refactoring file: {args.input}")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Read source code
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return 1
    
    # Initialize refactoring engine
    try:
        engine = RefactoringEngine(args.config)
        if args.model_dir:
            engine.load_pretrained_models(args.model_dir)
    except Exception as e:
        logger.error(f"Error initializing refactoring engine: {e}")
        return 1
    
    # Perform refactoring
    try:
        result = engine.refactor_code(
            source_code=source_code,
            optimization_target=args.target,
            use_genetic_optimization=not args.no_genetic,
            max_generations=args.generations
        )
    except Exception as e:
        logger.error(f"Error during refactoring: {e}")
        return 1
    
    # Save refactored code
    output_file = args.output or args.input.replace('.py', '_refactored.py')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result["refactored_code"])
        logger.info(f"Refactored code saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving refactored code: {e}")
        return 1
    
    # Save detailed results
    if args.save_results:
        result_file = output_file.replace('.py', '_results.json')
        try:
            serializable_result = {
                k: v for k, v in result.items() 
                if k not in ['original_code', 'refactored_code']
            }
            with open(result_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            logger.info(f"Detailed results saved to: {result_file}")
        except Exception as e:
            logger.warning(f"Error saving results: {e}")
    
    # Print summary
    print_refactoring_summary(result, args.input, output_file)
    
    return 0

def refactor_project(args):
    """Refactor an entire project."""
    logger.info(f"Refactoring project: {args.project}")
    
    # Find Python files
    python_files = []
    project_path = Path(args.project)
    
    if project_path.is_file() and project_path.suffix == '.py':
        python_files = [str(project_path)]
    else:
        # Search for Python files
        patterns = ['**/*.py']
        if args.include_tests:
            patterns.extend(['**/test_*.py', '**/tests/*.py'])
        
        for pattern in patterns:
            python_files.extend(glob.glob(str(project_path / pattern), recursive=True))
    
    # Filter files
    if args.exclude:
        exclude_patterns = args.exclude.split(',')
        filtered_files = []
        for file_path in python_files:
            if not any(pattern in file_path for pattern in exclude_patterns):
                filtered_files.append(file_path)
        python_files = filtered_files
    
    if not python_files:
        logger.error("No Python files found to refactor")
        return 1
    
    logger.info(f"Found {len(python_files)} Python files to refactor")
    
    # Initialize refactoring engine
    try:
        engine = RefactoringEngine(args.config)
        if args.model_dir:
            engine.load_pretrained_models(args.model_dir)
    except Exception as e:
        logger.error(f"Error initializing refactoring engine: {e}")
        return 1
    
    # Setup output directory
    output_dir = args.output or f"{args.project}_refactored"
    os.makedirs(output_dir, exist_ok=True)
    
    # Refactor files
    results = {
        "processed_files": [],
        "failed_files": [],
        "summary": {}
    }
    
    for i, file_path in enumerate(python_files):
        logger.info(f"Processing file {i+1}/{len(python_files)}: {file_path}")
        
        try:
            # Read source code
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Refactor code
            result = engine.refactor_code(
                source_code=source_code,
                optimization_target=args.target,
                use_genetic_optimization=not args.no_genetic,
                max_generations=args.generations
            )
            
            # Save refactored code
            rel_path = os.path.relpath(file_path, args.project)
            output_file = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result["refactored_code"])
            
            results["processed_files"].append({
                "input_file": file_path,
                "output_file": output_file,
                "confidence": result["confidence_score"],
                "pattern": result["refactoring_pattern"],
                "improvements": result.get("improvement_metrics", {})
            })
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            results["failed_files"].append({
                "file": file_path,
                "error": str(e)
            })
    
    # Generate summary
    if results["processed_files"]:
        confidences = [r["confidence"] for r in results["processed_files"]]
        results["summary"] = {
            "total_files": len(python_files),
            "processed_files": len(results["processed_files"]),
            "failed_files": len(results["failed_files"]),
            "success_rate": len(results["processed_files"]) / len(python_files),
            "average_confidence": sum(confidences) / len(confidences),
            "patterns_applied": list(set(r["pattern"] for r in results["processed_files"]))
        }
    
    # Save project results
    with open(os.path.join(output_dir, "refactoring_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_project_summary(results, output_dir)
    
    return 0

def evaluate_refactoring(args):
    """Evaluate refactoring results."""
    logger.info(f"Evaluating refactoring: {args.original} vs {args.refactored}")
    
    # Read files
    try:
        with open(args.original, 'r') as f:
            original_code = f.read()
        with open(args.refactored, 'r') as f:
            refactored_code = f.read()
    except Exception as e:
        logger.error(f"Error reading files: {e}")
        return 1
    
    # Initialize evaluator
    evaluator = RefactoringEvaluator({})
    
    # Perform evaluation
    try:
        results = evaluator.evaluate_refactoring(original_code, refactored_code)
        
        # Generate report
        report = evaluator.generate_evaluation_report(results)
        print(report)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to: {args.output}")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1
    
    return 0

def interactive_refactoring(args):
    """Interactive refactoring mode."""
    logger.info("Starting interactive refactoring mode")
    
    # Initialize refactoring engine
    try:
        engine = RefactoringEngine(args.config)
        if args.model_dir:
            engine.load_pretrained_models(args.model_dir)
    except Exception as e:
        logger.error(f"Error initializing refactoring engine: {e}")
        return 1
    
    print("Interactive Refactoring Mode")
    print("Enter 'quit' to exit, 'help' for commands")
    print()
    
    while True:
        try:
            # Get input
            command = input("refactor> ").strip()
            
            if command == 'quit':
                break
            elif command == 'help':
                print_help()
                continue
            elif command.startswith('file '):
                file_path = command[5:].strip()
                refactor_file_interactive(engine, file_path, args)
            elif command.startswith('code'):
                refactor_code_interactive(engine, args)
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    return 0

def refactor_file_interactive(engine, file_path: str, args):
    """Refactor a file in interactive mode."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        with open(file_path, 'r') as f:
            source_code = f.read()
        
        print(f"Refactoring {file_path}...")
        result = engine.refactor_code(
            source_code=source_code,
            optimization_target=args.target,
            use_genetic_optimization=not args.no_genetic
        )
        
        print(f"Pattern applied: {result['refactoring_pattern']}")
        print(f"Confidence: {result['confidence_score']:.3f}")
        
        # Ask if user wants to save
        save = input("Save refactored code? (y/n): ").strip().lower()
        if save == 'y':
            output_file = file_path.replace('.py', '_refactored.py')
            with open(output_file, 'w') as f:
                f.write(result["refactored_code"])
            print(f"Saved to: {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

def refactor_code_interactive(engine, args):
    """Refactor code snippet in interactive mode."""
    print("Enter code to refactor (end with '###' on a new line):")
    
    lines = []
    while True:
        line = input()
        if line.strip() == '###':
            break
        lines.append(line)
    
    source_code = '\n'.join(lines)
    
    if not source_code.strip():
        print("No code entered.")
        return
    
    try:
        print("Refactoring...")
        result = engine.refactor_code(
            source_code=source_code,
            optimization_target=args.target,
            use_genetic_optimization=not args.no_genetic
        )
        
        print("\nOriginal Code:")
        print("=" * 40)
        print(source_code)
        print("\nRefactored Code:")
        print("=" * 40)
        print(result["refactored_code"])
        print("\nResults:")
        print(f"Pattern: {result['refactoring_pattern']}")
        print(f"Confidence: {result['confidence_score']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")

def print_help():
    """Print help for interactive mode."""
    print("""
Available commands:
  file <path>     - Refactor a file
  code            - Refactor code snippet
  help            - Show this help
  quit            - Exit interactive mode
""")

def print_refactoring_summary(result: Dict, input_file: str, output_file: str):
    """Print refactoring summary."""
    print("\n" + "=" * 60)
    print("REFACTORING SUMMARY")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Pattern applied: {result['refactoring_pattern']}")
    print(f"Confidence: {result['confidence_score']:.3f}")
    print(f"Optimization target: {result['optimization_target']}")
    
    if result.get('candidates_found'):
        print(f"\nRefactoring opportunities found: {len(result['candidates_found'])}")
        for i, candidate in enumerate(result['candidates_found'][:3], 1):
            print(f"  {i}. {candidate['type']} at {candidate['location']}")
    
    if result.get('improvement_metrics'):
        print("\nImprovements:")
        for metric, value in result['improvement_metrics'].items():
            if 'percent' in metric and abs(value) > 1:
                direction = "improved" if value > 0 else "reduced"
                print(f"  {metric}: {abs(value):.1f}% {direction}")
    
    if result.get('genetic_optimization_used'):
        print(f"\nGenetic optimization: Enabled")
        if 'optimization_details' in result:
            details = result['optimization_details']
            print(f"  Generations: {details.get('generations_completed', 'N/A')}")
            print(f"  Pareto front size: {details.get('pareto_front_size', 'N/A')}")
    else:
        print(f"\nGenetic optimization: Disabled")

def print_project_summary(results: Dict, output_dir: str):
    """Print project refactoring summary."""
    summary = results.get("summary", {})
    
    print("\n" + "=" * 60)
    print("PROJECT REFACTORING SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total files: {summary.get('total_files', 0)}")
    print(f"Successfully processed: {summary.get('processed_files', 0)}")
    print(f"Failed: {summary.get('failed_files', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.2%}")
    
    if summary.get('average_confidence'):
        print(f"Average confidence: {summary['average_confidence']:.3f}")
    
    if summary.get('patterns_applied'):
        print(f"\nPatterns applied:")
        for pattern in summary['patterns_applied']:
            count = sum(1 for r in results['processed_files'] if r['pattern'] == pattern)
            print(f"  {pattern}: {count} files")
    
    if results.get('failed_files'):
        print(f"\nFailed files:")
        for failed in results['failed_files'][:5]:  # Show first 5 failures
            print(f"  {failed['file']}: {failed['error']}")
        if len(results['failed_files']) > 5:
            print(f"  ... and {len(results['failed_files']) - 5} more")

def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    if args.command == 'file':
        if not args.input:
            errors.append("Input file is required for 'file' command")
        elif not os.path.exists(args.input):
            errors.append(f"Input file does not exist: {args.input}")
    
    elif args.command == 'project':
        if not args.project:
            errors.append("Project path is required for 'project' command")
        elif not os.path.exists(args.project):
            errors.append(f"Project path does not exist: {args.project}")
    
    elif args.command == 'evaluate':
        if not args.original or not args.refactored:
            errors.append("Both original and refactored files are required for 'evaluate' command")
        elif not os.path.exists(args.original):
            errors.append(f"Original file does not exist: {args.original}")
        elif not os.path.exists(args.refactored):
            errors.append(f"Refactored file does not exist: {args.refactored}")
    
    if args.config and not os.path.exists(args.config):
        errors.append(f"Config file does not exist: {args.config}")
    
    if args.model_dir and not os.path.exists(args.model_dir):
        errors.append(f"Model directory does not exist: {args.model_dir}")
    
    if args.target not in ['balanced', 'quality', 'readability', 'performance', 'maintainability']:
        errors.append(f"Invalid optimization target: {args.target}")
    
    return errors

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AI-Powered Code Refactoring Tool")
    
    # Global arguments
    parser.add_argument("--config", "-c", default="config/model_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--model-dir", "-m", help="Directory containing pre-trained models")
    parser.add_argument("--target", "-t", default="balanced",
                       choices=["balanced", "quality", "readability", "performance", "maintainability"],
                       help="Optimization target")
    parser.add_argument("--no-genetic", action="store_true",
                       help="Disable genetic algorithm optimization")
    parser.add_argument("--generations", "-g", type=int, default=25,
                       help="Maximum generations for genetic algorithm")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # File refactoring
    file_parser = subparsers.add_parser("file", help="Refactor a single file")
    file_parser.add_argument("--input", "-i", required=True, help="Input Python file")
    file_parser.add_argument("--output", "-o", help="Output file (default: input_refactored.py)")
    file_parser.add_argument("--save-results", action="store_true",
                           help="Save detailed results to JSON file")
    
    # Project refactoring
    project_parser = subparsers.add_parser("project", help="Refactor an entire project")
    project_parser.add_argument("--project", "-p", required=True,
                               help="Project directory or file")
    project_parser.add_argument("--output", "-o", help="Output directory")
    project_parser.add_argument("--exclude", help="Comma-separated patterns to exclude")
    project_parser.add_argument("--include-tests", action="store_true",
                              help="Include test files")
    
    # Evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate refactoring results")
    eval_parser.add_argument("--original", "-orig", required=True,
                           help="Original code file")
    eval_parser.add_argument("--refactored", "-ref", required=True,
                           help="Refactored code file")
    eval_parser.add_argument("--output", "-o", help="Output evaluation report file")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive refactoring mode")
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    setup_logging(args.verbose)
    
    # Validate arguments
    errors = validate_arguments(args)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 1
    
    # Execute command
    try:
        if args.command == "file":
            return refactor_single_file(args)
        elif args.command == "project":
            return refactor_project(args)
        elif args.command == "evaluate":
            return evaluate_refactoring(args)
        elif args.command == "interactive":
            return interactive_refactoring(args)
        else:
            parser.print_help()
            return 0
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())