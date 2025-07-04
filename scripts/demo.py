# scripts/demo.py
"""
Demo script showing how to use the AI-Powered Code Refactoring Framework.
This script demonstrates various use cases and capabilities of the framework.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from refactoring.refactoring_engine import RefactoringEngine
from data.synthetic_generator import SyntheticDataGenerator
from utils.evaluation import RefactoringEvaluator

def demo_basic_refactoring():
    """Demonstrate basic code refactoring functionality."""
    
    print("=" * 60)
    print("DEMO 1: Basic Code Refactoring")
    print("=" * 60)
    
    # Sample code with refactoring opportunities
    original_code = """
def calculate_total_price(items, tax_rate, discount_rate):
    # Long method that needs refactoring
    total = 0
    for item in items:
        if item['price'] > 100:
            discounted_price = item['price'] * (1 - discount_rate)
            if discounted_price > 50:
                final_price = discounted_price * (1 + tax_rate)
                total += final_price
            else:
                final_price = discounted_price * (1 + tax_rate * 0.5)
                total += final_price
        else:
            final_price = item['price'] * (1 + tax_rate)
            total += final_price
    
    # Magic number
    if total > 1000:
        total = total * 0.95
    
    return total
"""
    
    print("Original Code:")
    print(original_code)
    print("\n" + "-" * 60 + "\n")
    
    # Initialize refactoring engine
    engine = RefactoringEngine()
    
    # Refactor the code
    result = engine.refactor_code(
        source_code=original_code,
        optimization_target="readability",
        use_genetic_optimization=True,
        max_generations=10
    )
    
    print("Refactored Code:")
    print(result["refactored_code"])
    print("\n" + "-" * 60 + "\n")
    
    print("Refactoring Results:")
    print(f"Pattern Applied: {result['refactoring_pattern']}")
    print(f"Confidence Score: {result['confidence_score']:.3f}")
    print(f"Optimization Target: {result['optimization_target']}")
    
    if result['improvement_metrics']:
        print("\nImprovements:")
        for metric, value in result['improvement_metrics'].items():
            if 'percent' in metric:
                print(f"  {metric}: {value:.1f}%")
    
    print("\n")

def demo_genetic_optimization():
    """Demonstrate genetic algorithm optimization."""
    
    print("=" * 60)
    print("DEMO 2: Genetic Algorithm Optimization")
    print("=" * 60)
    
    # Complex code example
    complex_code = """
class DataProcessor:
    def __init__(self):
        self.data = []
        self.processed_data = []
        self.errors = []
        self.config = {}
        
    def process_data(self, input_data, validation_rules, transformation_rules):
        for item in input_data:
            # Validation
            if item is None:
                self.errors.append("Null item found")
                continue
            if not isinstance(item, dict):
                self.errors.append("Invalid item type")
                continue
            if 'id' not in item:
                self.errors.append("Missing ID field")
                continue
            if 'value' not in item:
                self.errors.append("Missing value field")
                continue
            
            # Transformation
            processed_item = {}
            processed_item['id'] = item['id']
            
            if item['value'] > 0:
                if item['value'] < 100:
                    processed_item['category'] = 'low'
                    processed_item['processed_value'] = item['value'] * 1.1
                elif item['value'] < 1000:
                    processed_item['category'] = 'medium'
                    processed_item['processed_value'] = item['value'] * 1.05
                else:
                    processed_item['category'] = 'high'
                    processed_item['processed_value'] = item['value'] * 1.02
            else:
                processed_item['category'] = 'invalid'
                processed_item['processed_value'] = 0
                
            self.processed_data.append(processed_item)
        
        return self.processed_data
"""
    
    print("Complex Code for Optimization:")
    print(complex_code)
    print("\n" + "-" * 60 + "\n")
    
    engine = RefactoringEngine()
    
    # Compare different optimization targets
    targets = ["balanced", "quality", "readability", "maintainability"]
    
    for target in targets:
        print(f"Optimizing for: {target}")
        
        result = engine.refactor_code(
            source_code=complex_code,
            optimization_target=target,
            use_genetic_optimization=True,
            max_generations=15
        )
        
        print(f"  Confidence: {result['confidence_score']:.3f}")
        print(f"  Pattern: {result['refactoring_pattern']}")
        
        if 'optimization_details' in result:
            details = result['optimization_details']
            print(f"  Pareto Front Size: {details.get('pareto_front_size', 'N/A')}")
            print(f"  Generations: {details.get('generations_completed', 'N/A')}")
        
        print()

def demo_batch_processing():
    """Demonstrate batch refactoring of multiple files."""
    
    print("=" * 60)
    print("DEMO 3: Batch Processing")
    print("=" * 60)
    
    # Create sample files for batch processing
    sample_files = {
        "sample1.py": """
def calculate_area(width, height):
    if width > 0 and height > 0:
        if width > 10 and height > 10:
            return width * height * 0.9
        else:
            return width * height
    else:
        return 0
""",
        "sample2.py": """
class Calculator:
    def add(self, a, b):
        result = a + b
        if result > 100:
            result = result * 1.1
        return result
    
    def subtract(self, a, b):
        result = a - b
        if result < 0:
            result = 0
        return result
""",
        "sample3.py": """
def process_list(items):
    processed = []
    for item in items:
        if item > 10:
            if item < 100:
                processed.append(item * 2)
            else:
                processed.append(item * 1.5)
        else:
            processed.append(item)
    return processed
"""
    }
    
    # Create temporary directory and files
    temp_dir = "temp_batch_demo"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    for filename, content in sample_files.items():
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        file_paths.append(file_path)
    
    print(f"Created {len(file_paths)} sample files in {temp_dir}/")
    
    # Perform batch refactoring
    engine = RefactoringEngine()
    
    results = engine.batch_refactor(
        code_files=file_paths,
        output_directory="temp_batch_demo/refactored",
        optimization_target="balanced",
        use_genetic_optimization=False  # Faster for demo
    )
    
    print("\nBatch Processing Results:")
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {len(results['processed_files'])}")
    print(f"Failed: {len(results['failed_files'])}")
    
    if results['summary_statistics']:
        stats = results['summary_statistics']
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average confidence: {stats['average_confidence']:.3f}")
        print(f"Most common pattern: {stats['most_common_pattern']}")
    
    print("\nProcessed Files:")
    for file_result in results['processed_files']:
        print(f"  {file_result['input_file']} -> {file_result['output_file']}")
        print(f"    Pattern: {file_result['pattern']}")
        print(f"    Confidence: {file_result['confidence']:.3f}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temporary directory: {temp_dir}")

def demo_synthetic_data_generation():
    """Demonstrate synthetic dataset generation."""
    
    print("=" * 60)
    print("DEMO 4: Synthetic Data Generation")
    print("=" * 60)
    
    # Generate synthetic dataset
    config = {
        "dataset_size": 100,
        "output_dir": "temp_synthetic_demo"
    }
    
    generator = SyntheticDataGenerator(config)
    
    print("Generating synthetic refactoring patterns...")
    patterns = generator.generate_dataset(config["dataset_size"])
    
    print(f"Generated {len(patterns)} patterns")
    
    # Show pattern distribution
    pattern_counts = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    print("\nPattern Distribution:")
    for pattern_type, count in pattern_counts.items():
        print(f"  {pattern_type}: {count}")
    
    # Show sample patterns
    print("\nSample Patterns:")
    for i, pattern in enumerate(patterns[:3]):
        print(f"\nPattern {i+1}: {pattern.pattern_type}")
        print("Original Code:")
        print(pattern.original_code[:200] + "..." if len(pattern.original_code) > 200 else pattern.original_code)
        print("\nRefactored Code:")
        print(pattern.refactored_code[:200] + "..." if len(pattern.refactored_code) > 200 else pattern.refactored_code)
        print(f"Quality Improvement: {pattern.quality_improvement}")
        print("-" * 40)
    
    # Save dataset
    generator.save_dataset(patterns, config["output_dir"])
    print(f"\nDataset saved to {config['output_dir']}")
    
    # Cleanup
    import shutil
    if os.path.exists(config["output_dir"]):
        shutil.rmtree(config["output_dir"])
        print(f"Cleaned up: {config['output_dir']}")

def demo_evaluation_metrics():
    """Demonstrate evaluation capabilities."""
    
    print("=" * 60)
    print("DEMO 5: Evaluation Metrics")
    print("=" * 60)
    
    # Example of before and after refactoring
    original_code = """
def calculate_discount(price, customer_type, is_weekend):
    if customer_type == "premium":
        if is_weekend:
            if price > 100:
                return price * 0.8
            else:
                return price * 0.9
        else:
            if price > 100:
                return price * 0.85
            else:
                return price * 0.95
    else:
        if is_weekend:
            if price > 100:
                return price * 0.9
            else:
                return price * 0.95
        else:
            return price
"""
    
    refactored_code = """
# Constants for discount rates
PREMIUM_WEEKEND_HIGH = 0.8
PREMIUM_WEEKEND_LOW = 0.9
PREMIUM_WEEKDAY_HIGH = 0.85
PREMIUM_WEEKDAY_LOW = 0.95
REGULAR_WEEKEND_HIGH = 0.9
REGULAR_WEEKEND_LOW = 0.95
HIGH_PRICE_THRESHOLD = 100

def calculate_discount(price, customer_type, is_weekend):
    \"\"\"Calculate discount based on customer type, day, and price.\"\"\"
    
    if customer_type == "premium":
        return _calculate_premium_discount(price, is_weekend)
    else:
        return _calculate_regular_discount(price, is_weekend)

def _calculate_premium_discount(price, is_weekend):
    \"\"\"Calculate discount for premium customers.\"\"\"
    if is_weekend:
        return price * (PREMIUM_WEEKEND_HIGH if price > HIGH_PRICE_THRESHOLD else PREMIUM_WEEKEND_LOW)
    else:
        return price * (PREMIUM_WEEKDAY_HIGH if price > HIGH_PRICE_THRESHOLD else PREMIUM_WEEKDAY_LOW)

def _calculate_regular_discount(price, is_weekend):
    \"\"\"Calculate discount for regular customers.\"\"\"
    if is_weekend:
        return price * (REGULAR_WEEKEND_HIGH if price > HIGH_PRICE_THRESHOLD else REGULAR_WEEKEND_LOW)
    else:
        return price
"""
    
    print("Evaluating refactoring quality...")
    print("\nOriginal Code:")
    print(original_code)
    print("\n" + "-" * 60 + "\n")
    print("Refactored Code:")
    print(refactored_code)
    print("\n" + "-" * 60 + "\n")
    
    # Initialize evaluator
    config = {}
    evaluator = RefactoringEvaluator(config)
    
    # Perform comprehensive evaluation
    evaluation_results = evaluator.evaluate_refactoring(
        original_code=original_code,
        refactored_code=refactored_code
    )
    
    # Generate and display report
    report = evaluator.generate_evaluation_report(evaluation_results)
    print(report)
    
    # Show detailed metrics
    print("\nDetailed Metrics:")
    
    if 'code_quality' in evaluation_results:
        quality = evaluation_results['code_quality']
        print("\nCode Quality Metrics:")
        print("Original:")
        for metric, value in quality['original_metrics'].items():
            print(f"  {metric}: {value}")
        print("Refactored:")
        for metric, value in quality['refactored_metrics'].items():
            print(f"  {metric}: {value}")
    
    if 'readability' in evaluation_results:
        readability = evaluation_results['readability']
        print(f"\nReadability Score: {readability['readability_score']:.3f}")
        
    if 'maintainability' in evaluation_results:
        maintainability = evaluation_results['maintainability']
        print(f"Maintainability Score: {maintainability['maintainability_score']:.3f}")

def demo_model_comparison():
    """Demonstrate comparison between different models."""
    
    print("=" * 60)
    print("DEMO 6: Model Comparison")
    print("=" * 60)
    
    # Sample code for comparison
    test_code = """
def process_orders(orders):
    total = 0
    processed_orders = []
    
    for order in orders:
        if order['status'] == 'pending':
            if order['amount'] > 1000:
                order['priority'] = 'high'
                order['processed_amount'] = order['amount'] * 0.95
            elif order['amount'] > 100:
                order['priority'] = 'medium'
                order['processed_amount'] = order['amount'] * 0.98
            else:
                order['priority'] = 'low'
                order['processed_amount'] = order['amount']
            
            order['status'] = 'processed'
            total += order['processed_amount']
            processed_orders.append(order)
    
    return processed_orders, total
"""
    
    print("Test Code:")
    print(test_code)
    print("\n" + "-" * 60 + "\n")
    
    engine = RefactoringEngine()
    
    # Test different configurations
    configurations = [
        {"use_genetic_optimization": False, "name": "Ensemble Only"},
        {"use_genetic_optimization": True, "max_generations": 10, "name": "With Genetic Algorithm (10 gen)"},
        {"use_genetic_optimization": True, "max_generations": 25, "name": "With Genetic Algorithm (25 gen)"}
    ]
    
    results = []
    
    for config in configurations:
        print(f"Testing: {config['name']}")
        
        result = engine.refactor_code(
            source_code=test_code,
            optimization_target="balanced",
            use_genetic_optimization=config.get("use_genetic_optimization", False),
            max_generations=config.get("max_generations")
        )
        
        results.append({
            "config": config,
            "result": result
        })
        
        print(f"  Confidence: {result['confidence_score']:.3f}")
        print(f"  Pattern: {result['refactoring_pattern']}")
        
        if 'optimization_details' in result:
            details = result['optimization_details']
            if 'pareto_front_size' in details:
                print(f"  Pareto Front Size: {details['pareto_front_size']}")
            if 'generations_completed' in details:
                print(f"  Generations: {details['generations_completed']}")
        
        print()
    
    # Compare results
    print("Comparison Summary:")
    print("-" * 40)
    
    best_confidence = max(r['result']['confidence_score'] for r in results)
    
    for result_data in results:
        config = result_data['config']
        result = result_data['result']
        
        confidence = result['confidence_score']
        is_best = confidence == best_confidence
        
        print(f"{config['name']}: {confidence:.3f} {'(BEST)' if is_best else ''}")

def demo_real_world_example():
    """Demonstrate refactoring of a real-world-like example."""
    
    print("=" * 60)
    print("DEMO 7: Real-World Example")
    print("=" * 60)
    
    # Complex real-world-like code
    real_world_code = """
import json
import csv
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        self.data = []
        self.config = {}
        
    def generate_sales_report(self, sales_data, config_file, output_format):
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Process sales data
        processed_data = []
        total_sales = 0
        
        for sale in sales_data:
            # Validation
            if not sale:
                continue
            if 'amount' not in sale or 'date' not in sale:
                continue
            if sale['amount'] <= 0:
                continue
                
            # Date processing
            sale_date = datetime.strptime(sale['date'], '%Y-%m-%d')
            current_year = datetime.now().year
            
            if sale_date.year != current_year:
                continue
                
            # Calculate commission
            if sale['amount'] > 10000:
                commission = sale['amount'] * 0.05
            elif sale['amount'] > 5000:
                commission = sale['amount'] * 0.03
            elif sale['amount'] > 1000:
                commission = sale['amount'] * 0.02
            else:
                commission = sale['amount'] * 0.01
                
            # Apply regional multiplier
            if sale.get('region') == 'north':
                commission *= 1.1
            elif sale.get('region') == 'south':
                commission *= 1.05
            elif sale.get('region') == 'east':
                commission *= 1.08
            elif sale.get('region') == 'west':
                commission *= 1.03
                
            sale['commission'] = commission
            total_sales += sale['amount']
            processed_data.append(sale)
        
        # Generate output
        if output_format == 'json':
            output = json.dumps(processed_data, indent=2)
        elif output_format == 'csv':
            output = self._generate_csv_output(processed_data)
        else:
            output = str(processed_data)
            
        return output, total_sales
    
    def _generate_csv_output(self, data):
        import io
        output = io.StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return output.getvalue()
"""
    
    print("Real-World Example Code:")
    print(real_world_code)
    print("\n" + "-" * 60 + "\n")
    
    print("Refactoring with focus on maintainability...")
    
    engine = RefactoringEngine()
    
    result = engine.refactor_code(
        source_code=real_world_code,
        optimization_target="maintainability",
        use_genetic_optimization=True,
        max_generations=20
    )
    
    print("Refactored Code:")
    print(result["refactored_code"])
    print("\n" + "-" * 60 + "\n")
    
    print("Analysis:")
    print(f"Confidence Score: {result['confidence_score']:.3f}")
    print(f"Applied Pattern: {result['refactoring_pattern']}")
    
    if result['candidates_found']:
        print(f"\nRefactoring Opportunities Found:")
        for i, candidate in enumerate(result['candidates_found'][:5], 1):
            print(f"  {i}. {candidate['type']} at {candidate['location']}")
            print(f"     Reason: {candidate['reason']}")
            print(f"     Priority: {candidate['priority']:.1f}")
    
    if result['improvement_metrics']:
        print(f"\nImprovements:")
        for metric, value in result['improvement_metrics'].items():
            if 'percent' in metric and abs(value) > 1:
                direction = "improved" if value > 0 else "reduced"
                print(f"  {metric}: {abs(value):.1f}% {direction}")

def main():
    """Run all demo functions."""
    
    print("AI-Powered Code Refactoring Framework - Demo")
    print("=" * 60)
    
    demos = [
        ("Basic Refactoring", demo_basic_refactoring),
        ("Genetic Optimization", demo_genetic_optimization),
        ("Batch Processing", demo_batch_processing),
        ("Synthetic Data Generation", demo_synthetic_data_generation),
        ("Evaluation Metrics", demo_evaluation_metrics),
        ("Model Comparison", demo_model_comparison),
        ("Real-World Example", demo_real_world_example)
    ]
    
    print("Available demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all demos...")
    print()
    
    for name, demo_func in demos:
        try:
            demo_func()
            print("\n" + "✓" * 60)
            print(f"Demo '{name}' completed successfully!")
            print("✓" * 60)
            print("\n")
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
            print()
    
    print("All demos completed!")
    print()
    print("Next Steps:")
    print("1. Install the framework: pip install -e .")
    print("2. Generate training data: python scripts/generate_dataset.py")
    print("3. Train models: python scripts/train_models.py")
    print("4. Use the refactoring engine:")
    print("   python -m refactoring.refactoring_engine --input your_code.py")
    print()
    print("For more information, see the README.md file.")

if __name__ == "__main__":
    main()