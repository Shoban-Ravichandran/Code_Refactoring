"""
Unit tests for refactoring engine and related components.
"""

import unittest
import tempfile
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from refactoring.refactoring_engine import RefactoringEngine
from refactoring.code_analyzer import CodeAnalyzer, RefactoringOpportunity, QualityIssue
from refactoring.quality_metrics import QualityMetrics, MetricResult, calculate_quality_score
from utils.evaluation import RefactoringEvaluator
from utils.code_utils import CodeProcessor, similarity_score
from utils.ast_utils import ASTAnalyzer, ASTTransformer

class TestRefactoringEngine(unittest.TestCase):
    """Test cases for the main refactoring engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_code = """
def calculate_price(items, tax_rate, discount_rate):
    total = 0
    for item in items:
        if item['price'] > 100:
            discounted_price = item['price'] * (1 - discount_rate)
            total += discounted_price * (1 + tax_rate)
        else:
            total += item['price'] * (1 + tax_rate)
    
    if total > 1000:
        total = total * 0.95
    
    return total
"""
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('refactoring.refactoring_engine.EnsembleRefactoringModel')
    @patch('refactoring.refactoring_engine.NSGA2')
    @patch('refactoring.refactoring_engine.RefactoringEvaluator')
    def test_engine_initialization(self, mock_evaluator, mock_nsga2, mock_ensemble):
        """Test refactoring engine initialization."""
        # Create a temporary config file
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        config_content = """
model:
  ensemble:
    codebert_weight: 0.4
    starcoder_weight: 0.6
genetic_algorithm:
  nsga2:
    population_size: 20
    generations: 10
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Initialize engine
        engine = RefactoringEngine(config_path)
        
        # Assertions
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.config)
        mock_ensemble.assert_called_once()
        mock_nsga2.assert_called_once()
        mock_evaluator.assert_called_once()
    
    @patch('refactoring.refactoring_engine.EnsembleRefactoringModel')
    @patch('refactoring.refactoring_engine.NSGA2')
    def test_code_refactoring_without_genetic(self, mock_nsga2, mock_ensemble):
        """Test code refactoring without genetic optimization."""
        # Setup mocks
        mock_ensemble_instance = Mock()
        mock_ensemble.return_value = mock_ensemble_instance
        
        mock_ensemble_instance.generate_refactored_code.return_value = {
            "refactored_code": "def improved_code(): pass",
            "ensemble_confidence": 0.8,
            "selected_model": "codebert",
            "quality_scores": {"readability": 0.9}
        }
        
        # Initialize engine
        engine = RefactoringEngine()
        engine.ensemble_model = mock_ensemble_instance
        
        # Mock evaluator
        engine.evaluator = Mock()
        engine.evaluator.evaluate_refactoring.return_value = {
            "syntax_correctness": {"is_valid": True, "score": 1.0}
        }
        
        # Perform refactoring
        result = engine.refactor_code(
            source_code=self.sample_code,
            use_genetic_optimization=False
        )
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('refactored_code', result)
        self.assertIn('confidence_score', result)
        self.assertIn('refactoring_pattern', result)
        self.assertFalse(result['genetic_optimization_used'])
    
    @patch('refactoring.refactoring_engine.EnsembleRefactoringModel')
    @patch('refactoring.refactoring_engine.NSGA2')
    @patch('refactoring.refactoring_engine.create_objective_functions')
    def test_code_refactoring_with_genetic(self, mock_objectives, mock_nsga2, mock_ensemble):
        """Test code refactoring with genetic optimization."""
        # Setup mocks
        mock_ensemble_instance = Mock()
        mock_nsga2_instance = Mock()
        mock_ensemble.return_value = mock_ensemble_instance
        mock_nsga2.return_value = mock_nsga2_instance
        
        # Mock genetic algorithm results
        from genetic_algorithm.nsga2 import Individual
        best_individual = Individual(
            genes=[0.7, 0.8, 0.6, 0.5, 0.9, 0.4, 0.7, 0.8],
            objectives=[0.8, 0.7, 0.6, 0.9],
            fitness=0.75
        )
        best_individual.refactored_code = "def optimized_code(): return 42"
        best_individual.pattern_type = "extract_method"
        
        mock_nsga2_instance.evolve.return_value = (
            [best_individual],
            {
                "total_generations": 15,
                "final_hypervolume": 0.85,
                "pareto_front_size": 3
            }
        )
        
        mock_objectives.return_value = [Mock(), Mock(), Mock(), Mock()]
        
        # Initialize engine
        engine = RefactoringEngine()
        engine.ensemble_model = mock_ensemble_instance
        engine.genetic_algorithm = mock_nsga2_instance
        
        # Mock evaluator
        engine.evaluator = Mock()
        engine.evaluator.evaluate_refactoring.return_value = {
            "syntax_correctness": {"is_valid": True, "score": 1.0}
        }
        
        # Perform refactoring
        result = engine.refactor_code(
            source_code=self.sample_code,
            use_genetic_optimization=True,
            max_generations=15
        )
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('refactored_code', result)
        self.assertIn('optimization_details', result)
        self.assertTrue(result['genetic_optimization_used'])
        self.assertIn('pareto_front_size', result['optimization_details'])
    
    def test_batch_refactoring(self):
        """Test batch refactoring functionality."""
        # Create test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"test_file_{i}.py")
            with open(file_path, 'w') as f:
                f.write(f"def test_function_{i}(): return {i}")
            test_files.append(file_path)
        
        # Mock the refactoring engine
        with patch.object(RefactoringEngine, 'refactor_code') as mock_refactor:
            mock_refactor.return_value = {
                'refactored_code': 'def improved(): pass',
                'confidence_score': 0.8,
                'refactoring_pattern': 'extract_method',
                'improvement_metrics': {}
            }
            
            engine = RefactoringEngine()
            results = engine.batch_refactor(
                code_files=test_files,
                output_directory=os.path.join(self.temp_dir, 'output')
            )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertIn('processed_files', results)
        self.assertIn('failed_files', results)
        self.assertIn('summary_statistics', results)
        self.assertEqual(len(results['processed_files']), 3)

class TestCodeAnalyzer(unittest.TestCase):
    """Test cases for code analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_code = """
class DataProcessor:
    def __init__(self):
        self.data = []
        self.processed_count = 0
    
    def process_large_dataset(self, items, threshold, multiplier, format_type):
        # This is a very long method that does too many things
        processed_items = []
        failed_items = []
        summary_stats = {'total': 0, 'processed': 0, 'failed': 0}
        
        for item in items:
            summary_stats['total'] += 1
            
            if item is None:
                failed_items.append(item)
                summary_stats['failed'] += 1
                continue
                
            if not isinstance(item, dict):
                failed_items.append(item)
                summary_stats['failed'] += 1
                continue
                
            if 'value' not in item:
                failed_items.append(item)
                summary_stats['failed'] += 1
                continue
                
            if item['value'] > threshold:
                if item['value'] < 1000:
                    processed_value = item['value'] * multiplier
                    if format_type == 'percentage':
                        processed_value = processed_value * 100
                    elif format_type == 'currency':
                        processed_value = f"${processed_value:.2f}"
                    
                    processed_item = {
                        'original': item,
                        'processed_value': processed_value,
                        'timestamp': 12345
                    }
                    processed_items.append(processed_item)
                    summary_stats['processed'] += 1
                    self.processed_count += 1
                else:
                    failed_items.append(item)
                    summary_stats['failed'] += 1
            else:
                failed_items.append(item)
                summary_stats['failed'] += 1
        
        self.data.extend(processed_items)
        return processed_items, failed_items, summary_stats
"""
        self.analyzer = CodeAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test code analyzer initialization."""
        self.assertIsInstance(self.analyzer, CodeAnalyzer)
        self.assertIsInstance(self.analyzer.thresholds, dict)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive code analysis."""
        results = self.analyzer.analyze_code(self.sample_code)
        
        self.assertIsInstance(results, dict)
        self.assertIn('refactoring_opportunities', results)
        self.assertIn('quality_issues', results)
        self.assertIn('code_metrics', results)
        self.assertIn('maintainability_score', results)
        
        # Check that opportunities were found
        opportunities = results['refactoring_opportunities']
        self.assertIsInstance(opportunities, list)
        self.assertGreater(len(opportunities), 0)
        
        # Check opportunity structure
        for opp in opportunities:
            self.assertIsInstance(opp, RefactoringOpportunity)
            self.assertIn(opp.type, [
                'extract_method', 'extract_variable', 'replace_magic_numbers',
                'simplify_conditional', 'remove_dead_code', 'extract_class'
            ])
    
    def test_extract_method_detection(self):
        """Test detection of extract method opportunities."""
        results = self.analyzer.analyze_code(self.sample_code)
        opportunities = results['refactoring_opportunities']
        
        # Should detect long method needing extraction
        extract_method_opps = [opp for opp in opportunities if opp.type == 'extract_method']
        self.assertGreater(len(extract_method_opps), 0)
        
        # Check opportunity details
        opp = extract_method_opps[0]
        self.assertIn('method_name', opp.details)
        self.assertGreater(opp.details['length'], self.analyzer.thresholds['method_length'])
    
    def test_quality_issues_detection(self):
        """Test detection of quality issues."""
        results = self.analyzer.analyze_code(self.sample_code)
        issues = results['quality_issues']
        
        self.assertIsInstance(issues, list)
        
        # Check issue structure
        for issue in issues:
            self.assertIsInstance(issue, QualityIssue)
            self.assertIn(issue.type, [
                'naming_convention', 'missing_documentation', 'performance', 'security'
            ])
            self.assertIn(issue.severity, ['low', 'medium', 'high', 'critical'])

class TestQualityMetrics(unittest.TestCase):
    """Test cases for quality metrics calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_code = """
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    \"\"\"A simple calculator class.\"\"\"
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        \"\"\"Add two numbers.\"\"\"
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        \"\"\"Multiply two numbers.\"\"\"
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
"""
        self.metrics_calculator = QualityMetrics()
    
    def test_metrics_initialization(self):
        """Test quality metrics calculator initialization."""
        self.assertIsInstance(self.metrics_calculator, QualityMetrics)
        self.assertIsInstance(self.metrics_calculator.metrics_registry, dict)
    
    def test_all_metrics_calculation(self):
        """Test calculation of all metrics."""
        results = self.metrics_calculator.calculate_all_metrics(self.sample_code)
        
        self.assertIsInstance(results, dict)
        
        # Check that all categories are present
        categories = set(metric.category for metric in results.values())
        expected_categories = {'complexity', 'size', 'maintainability', 'readability', 'design', 'documentation'}
        self.assertTrue(expected_categories.issubset(categories))
        
        # Check metric structure
        for metric in results.values():
            self.assertIsInstance(metric, MetricResult)
            self.assertIsInstance(metric.value, (int, float))
            self.assertIsInstance(metric.interpretation, str)
    
    def test_complexity_metrics(self):
        """Test complexity metrics calculation."""
        import ast
        tree = ast.parse(self.sample_code)
        results = self.metrics_calculator.calculate_complexity_metrics(self.sample_code, tree)
        
        self.assertIn('cyclomatic_complexity', results)
        self.assertIn('cognitive_complexity', results)
        self.assertIn('max_nesting_depth', results)
        
        # Check reasonable values
        cc = results['cyclomatic_complexity']
        self.assertGreaterEqual(cc.value, 1)  # At least base complexity
    
    def test_size_metrics(self):
        """Test size metrics calculation."""
        import ast
        tree = ast.parse(self.sample_code)
        results = self.metrics_calculator.calculate_size_metrics(self.sample_code, tree)
        
        self.assertIn('lines_of_code', results)
        self.assertIn('source_lines_of_code', results)
        self.assertIn('number_of_functions', results)
        self.assertIn('number_of_classes', results)
        
        # Check reasonable values
        self.assertGreater(results['lines_of_code'].value, 0)
        self.assertEqual(results['number_of_classes'].value, 1)
        self.assertGreaterEqual(results['number_of_functions'].value, 3)  # fibonacci, add, multiply
    
    def test_maintainability_metrics(self):
        """Test maintainability metrics calculation."""
        import ast
        tree = ast.parse(self.sample_code)
        results = self.metrics_calculator.calculate_maintainability_metrics(self.sample_code, tree)
        
        self.assertIn('maintainability_index', results)
        self.assertIn('coupling', results)
        self.assertIn('cohesion', results)
        
        # Check values are in reasonable ranges
        mi = results['maintainability_index']
        self.assertGreaterEqual(mi.value, 0)
        self.assertLessEqual(mi.value, 100)
    
    def test_metrics_comparison(self):
        """Test metrics comparison functionality."""
        import ast
        tree = ast.parse(self.sample_code)
        
        # Calculate metrics for original code
        metrics1 = self.metrics_calculator.calculate_all_metrics(self.sample_code)
        
        # Create slightly modified code
        modified_code = self.sample_code.replace("fibonacci", "fib")
        metrics2 = self.metrics_calculator.calculate_all_metrics(modified_code)
        
        # Compare metrics
        comparison = self.metrics_calculator.compare_metrics(metrics1, metrics2)
        
        self.assertIsInstance(comparison, dict)
        
        # Check comparison structure
        for metric_name, comp_data in comparison.items():
            self.assertIn('before', comp_data)
            self.assertIn('after', comp_data)
            self.assertIn('change', comp_data)
            self.assertIn('change_percent', comp_data)
            self.assertIn('improvement', comp_data)

class TestRefactoringEvaluator(unittest.TestCase):
    """Test cases for refactoring evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_code = """
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
        
        self.refactored_code = """
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
        
        self.evaluator = RefactoringEvaluator({})
    
    def test_evaluator_initialization(self):
        """Test refactoring evaluator initialization."""
        self.assertIsInstance(self.evaluator, RefactoringEvaluator)
        self.assertIsInstance(self.evaluator.metrics, list)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive refactoring evaluation."""
        results = self.evaluator.evaluate_refactoring(
            self.original_code, 
            self.refactored_code
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('syntax_correctness', results)
        self.assertIn('semantic_equivalence', results)
        self.assertIn('code_quality', results)
        self.assertIn('overall_score', results)
        
        # Check syntax correctness
        syntax = results['syntax_correctness']
        self.assertTrue(syntax['is_valid'])
        self.assertEqual(syntax['score'], 1.0)
    
    def test_syntax_correctness_evaluation(self):
        """Test syntax correctness evaluation."""
        # Test with valid code
        valid_result = self.evaluator.evaluate_syntax_correctness("def test(): pass")
        self.assertTrue(valid_result['is_valid'])
        self.assertEqual(valid_result['score'], 1.0)
        
        # Test with invalid code
        invalid_result = self.evaluator.evaluate_syntax_correctness("def test( pass")
        self.assertFalse(invalid_result['is_valid'])
        self.assertEqual(invalid_result['score'], 0.0)
        self.assertIsNotNone(invalid_result['error'])
    
    def test_semantic_equivalence_evaluation(self):
        """Test semantic equivalence evaluation."""
        results = self.evaluator.evaluate_semantic_equivalence(
            self.original_code, 
            self.refactored_code
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('semantic_score', results)
        self.assertIn('ast_similarity', results)
        self.assertIn('is_equivalent', results)
        
        # Check reasonable values
        self.assertGreaterEqual(results['semantic_score'], 0.0)
        self.assertLessEqual(results['semantic_score'], 1.0)
    
    def test_code_quality_improvement_evaluation(self):
        """Test code quality improvement evaluation."""
        results = self.evaluator.evaluate_code_quality_improvement(
            self.original_code,
            self.refactored_code
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('original_metrics', results)
        self.assertIn('refactored_metrics', results)
        self.assertIn('improvements', results)
        self.assertIn('overall_improvement', results)
    
    def test_evaluation_report_generation(self):
        """Test evaluation report generation."""
        results = self.evaluator.evaluate_refactoring(
            self.original_code,
            self.refactored_code
        )
        
        report = self.evaluator.generate_evaluation_report(results)
        
        self.assertIsInstance(report, str)
        self.assertIn('CODE REFACTORING EVALUATION REPORT', report)
        self.assertIn('Overall Refactoring Score', report)

class TestCodeUtilities(unittest.TestCase):
    """Test cases for code utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_code = """
import os
import sys
from typing import List, Dict

def process_data(items: List[Dict]) -> Dict:
    \"\"\"Process a list of data items.\"\"\"
    result = {'processed': 0, 'total': len(items)}
    
    for item in items:
        if item.get('valid', True):
            # Process valid item
            processed_item = {
                'id': item['id'],
                'value': item['value'] * 2,
                'status': 'processed'
            }
            result['processed'] += 1
        else:
            # Skip invalid item
            continue
    
    return result

unused_variable = "This is not used anywhere"
"""
        self.processor = CodeProcessor()
    
    def test_code_processor_initialization(self):
        """Test code processor initialization."""
        self.assertIsInstance(self.processor, CodeProcessor)
        self.assertIsInstance(self.processor.keywords, set)
    
    def test_code_normalization(self):
        """Test code normalization."""
        normalized = self.processor.normalize_code(self.sample_code)
        
        self.assertIsInstance(normalized, str)
        self.assertNotIn('#', normalized)  # Comments should be removed
    
    def test_function_extraction(self):
        """Test function extraction."""
        functions = self.processor.extract_functions(self.sample_code)
        
        self.assertIsInstance(functions, list)
        self.assertEqual(len(functions), 1)  # One function: process_data
        
        func = functions[0]
        self.assertEqual(func['name'], 'process_data')
        self.assertIsInstance(func['args'], list)
        self.assertIn('items', func['args'])
    
    def test_import_extraction(self):
        """Test import extraction."""
        imports = self.processor.extract_imports(self.sample_code)
        
        self.assertIsInstance(imports, dict)
        self.assertIn('standard', imports)
        self.assertIn('from_imports', imports)
        
        # Check that imports were found
        self.assertIn('os', imports['standard'])
        self.assertIn('sys', imports['standard'])
    
    def test_code_metrics_calculation(self):
        """Test code metrics calculation."""
        metrics = self.processor.get_code_metrics(self.sample_code)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_lines', metrics)
        self.assertIn('code_lines', metrics)
        self.assertIn('functions', metrics)
        self.assertIn('classes', metrics)
        
        # Check reasonable values
        self.assertGreater(metrics['total_lines'], 0)
        self.assertEqual(metrics['functions'], 1)
        self.assertEqual(metrics['classes'], 0)
    
    def test_code_smell_detection(self):
        """Test code smell detection."""
        smells = self.processor.detect_code_smells(self.sample_code)
        
        self.assertIsInstance(smells, list)
        
        # Check smell structure
        for smell in smells:
            self.assertIn('type', smell)
            self.assertIn('location', smell)
            self.assertIn('description', smell)
            self.assertIn('severity', smell)
    
    def test_similarity_calculation(self):
        """Test code similarity calculation."""
        code1 = "def test(): return 1"
        code2 = "def test(): return 2"
        code3 = "def different(): pass"
        
        # Similar codes should have high similarity
        sim1 = similarity_score(code1, code2)
        self.assertGreater(sim1, 0.5)
        
        # Different codes should have lower similarity
        sim2 = similarity_score(code1, code3)
        self.assertLess(sim2, sim1)

class TestASTUtilities(unittest.TestCase):
    """Test cases for AST utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_code = """
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result
    
    def multiply(self, a, b):
        return a * b
"""
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
    
    def test_ast_analyzer_initialization(self):
        """Test AST analyzer initialization."""
        self.assertIsInstance(self.analyzer, ASTAnalyzer)
        self.assertIsInstance(self.analyzer.node_types, dict)
    
    def test_code_parsing(self):
        """Test code parsing into AST."""
        tree = self.analyzer.parse_code(self.sample_code)
        
        self.assertIsNotNone(tree)
        import ast
        self.assertIsInstance(tree, ast.AST)
    
    def test_node_type_counting(self):
        """Test AST node type counting."""
        tree = self.analyzer.parse_code(self.sample_code)
        node_counts = self.analyzer.get_node_types(tree)
        
        self.assertIsInstance(node_counts, dict)
        self.assertIn('ClassDef', node_counts)
        self.assertIn('FunctionDef', node_counts)
        
        # Check reasonable counts
        self.assertEqual(node_counts['ClassDef'], 1)
        self.assertGreaterEqual(node_counts['FunctionDef'], 3)  # __init__, add, multiply
    
    def test_function_info_extraction(self):
        """Test function information extraction."""
        tree = self.analyzer.parse_code(self.sample_code)
        import ast
        functions = self.analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        
        self.assertGreater(len(functions), 0)
        
        func_info = self.analyzer.get_function_info(functions[0])
        self.assertIsInstance(func_info, dict)
        self.assertIn('name', func_info)
        self.assertIn('args', func_info)
        self.assertIn('complexity', func_info)
    
    def test_class_info_extraction(self):
        """Test class information extraction."""
        tree = self.analyzer.parse_code(self.sample_code)
        import ast
        classes = self.analyzer.find_nodes_by_type(tree, ast.ClassDef)
        
        self.assertEqual(len(classes), 1)
        
        class_info = self.analyzer.get_class_info(classes[0])
        self.assertIsInstance(class_info, dict)
        self.assertIn('name', class_info)
        self.assertIn('methods', class_info)
        self.assertIn('attributes', class_info)
        
        self.assertEqual(class_info['name'], 'Calculator')
    
    def test_complexity_calculation(self):
        """Test complexity calculation."""
        tree = self.analyzer.parse_code(self.sample_code)
        complexity = self.analyzer.calculate_complexity(tree)
        
        self.assertIsInstance(complexity, int)
        self.assertGreaterEqual(complexity, 1)
    
    def test_ast_transformation(self):
        """Test AST transformation capabilities."""
        tree = self.analyzer.parse_code("def test(): return 42")
        
        # Test rename transformation
        transformed = self.transformer.rename_identifier(tree, "test", "renamed_test")
        
        self.assertIsNotNone(transformed)
        # Would need to convert back to code to verify transformation

class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_refactoring_workflow(self):
        """Test complete refactoring workflow."""
        # Sample code with clear refactoring opportunities
        code_with_issues = """
def bad_function(x, y, z, a, b, c, d):
    # Long parameter list
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    # Deep nesting
                    result = x * y * z * a * b * c * d * 42  # Magic number
                    return result
                else:
                    return 0
            else:
                return 0
        else:
            return 0
    else:
        return 0
"""
        
        # Step 1: Analyze code
        analyzer = CodeAnalyzer()
        analysis = analyzer.analyze_code(code_with_issues)
        
        # Should find multiple issues
        self.assertGreater(len(analysis['refactoring_opportunities']), 0)
        
        # Step 2: Calculate quality metrics
        metrics_calc = QualityMetrics()
        original_metrics = metrics_calc.calculate_all_metrics(code_with_issues)
        
        # Should have poor complexity metrics
        cc = original_metrics['cyclomatic_complexity']
        self.assertGreater(cc.value, 5)  # High complexity
        
        # Step 3: Mock refactoring (would normally use engine)
        improved_code = """
MULTIPLIER_CONSTANT = 42

def calculate_product(numbers):
    \"\"\"Calculate product of positive numbers.\"\"\"
    if not all(num > 0 for num in numbers):
        return 0
    
    result = 1
    for num in numbers:
        result *= num
    
    return result * MULTIPLIER_CONSTANT
"""
        
        # Step 4: Evaluate improvement
        evaluator = RefactoringEvaluator({})
        evaluation = evaluator.evaluate_refactoring(code_with_issues, improved_code)
        
        # Should show improvements
        self.assertTrue(evaluation['syntax_correctness']['is_valid'])
        self.assertGreater(evaluation['overall_score'], 0.5)
    
    def test_error_handling_scenarios(self):
        """Test error handling in various scenarios."""
        # Test with invalid code
        invalid_code = "def broken_syntax( invalid"
        
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(invalid_code)
        
        # Should handle gracefully
        self.assertIn('error', result)
        
        # Test with empty code
        empty_result = analyzer.analyze_code("")
        self.assertIn('error', empty_result)
    
    def test_performance_with_large_code(self):
        """Test performance with larger code bases."""
        # Generate a larger code sample
        large_code = """
import os
import sys
from typing import List, Dict, Optional

class LargeClass:
    def __init__(self):
        self.data = []
        self.cache = {}
        self.stats = {'processed': 0, 'errors': 0}
    
"""
        
        # Add many methods
        for i in range(20):
            large_code += f"""
    def method_{i}(self, param1, param2):
        \"\"\"Method {i} documentation.\"\"\"
        if param1 > {i}:
            result = param1 * param2 + {i}
            self.stats['processed'] += 1
            return result
        else:
            self.stats['errors'] += 1
            return None
"""
        
        # Test that analysis completes in reasonable time
        import time
        start_time = time.time()
        
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(large_code)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 10 seconds
        self.assertLess(execution_time, 10.0)
        self.assertNotIn('error', result)

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)