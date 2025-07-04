# src/genetic_algorithm/fitness_functions.py
import ast
import numpy as np
from typing import Dict, List, Any
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from radon.raw import analyze
import logging
from .nsga2 import Individual
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class FitnessEvaluator:
    """Evaluates fitness objectives for code refactoring."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.objectives = config.get("genetic_algorithm", {}).get("objectives", [])
        
    def evaluate_code_quality(self, individual: Individual, code_analyzer) -> float:
        """Evaluate code quality objective."""
        try:
            # Get refactored code based on individual's genes
            refactored_code = self._apply_refactoring(individual, code_analyzer)
            
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(refactored_code)
            
            # Weighted quality score
            quality_score = (
                0.3 * metrics["maintainability_index"] +
                0.25 * (1.0 - metrics["complexity_normalized"]) +
                0.25 * metrics["readability_score"] +
                0.2 * metrics["documentation_ratio"]
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating code quality: {e}")
            return 0.0
    
    def evaluate_readability(self, individual: Individual, code_analyzer) -> float:
        """Evaluate code readability objective."""
        try:
            refactored_code = self._apply_refactoring(individual, code_analyzer)
            
            # Calculate readability metrics
            readability_score = 0.0
            
            # Parse AST for analysis
            tree = ast.parse(refactored_code)
            
            # Metrics that contribute to readability
            line_length_score = self._evaluate_line_length(refactored_code)
            naming_score = self._evaluate_naming_conventions(tree)
            complexity_score = self._evaluate_method_complexity(refactored_code)
            structure_score = self._evaluate_code_structure(tree)
            comment_score = self._evaluate_comments(refactored_code)
            
            readability_score = (
                0.2 * line_length_score +
                0.3 * naming_score +
                0.2 * complexity_score +
                0.2 * structure_score +
                0.1 * comment_score
            )
            
            return min(max(readability_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating readability: {e}")
            return 0.0
    
    def evaluate_performance(self, individual: Individual, code_analyzer) -> float:
        """Evaluate performance objective."""
        try:
            original_code = code_analyzer.original_code
            refactored_code = self._apply_refactoring(individual, code_analyzer)
            
            # Performance metrics
            performance_score = 0.0
            
            # Cyclomatic complexity reduction
            original_complexity = self._get_cyclomatic_complexity(original_code)
            refactored_complexity = self._get_cyclomatic_complexity(refactored_code)
            
            if original_complexity > 0:
                complexity_improvement = max(0, (original_complexity - refactored_complexity) / original_complexity)
            else:
                complexity_improvement = 0.0
            
            # Code size reduction (fewer lines can mean better performance)
            original_lines = len(original_code.strip().split('\n'))
            refactored_lines = len(refactored_code.strip().split('\n'))
            
            if original_lines > 0:
                size_improvement = max(0, (original_lines - refactored_lines) / original_lines)
            else:
                size_improvement = 0.0
            
            # Nested loop reduction
            original_nesting = self._count_nested_loops(original_code)
            refactored_nesting = self._count_nested_loops(refactored_code)
            nesting_improvement = max(0, (original_nesting - refactored_nesting) / max(1, original_nesting))
            
            # Method call optimization
            method_call_score = self._evaluate_method_calls(refactored_code)
            
            performance_score = (
                0.4 * complexity_improvement +
                0.2 * size_improvement +
                0.2 * nesting_improvement +
                0.2 * method_call_score
            )
            
            return min(max(performance_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating performance: {e}")
            return 0.0
    
    def evaluate_maintainability(self, individual: Individual, code_analyzer) -> float:
        """Evaluate maintainability objective."""
        try:
            refactored_code = self._apply_refactoring(individual, code_analyzer)
            
            # Maintainability metrics
            maintainability_score = 0.0
            
            # Method length
            method_length_score = self._evaluate_method_length(refactored_code)
            
            # Class cohesion
            cohesion_score = self._evaluate_class_cohesion(refactored_code)
            
            # Coupling
            coupling_score = self._evaluate_coupling(refactored_code)
            
            # Code duplication
            duplication_score = self._evaluate_code_duplication(refactored_code)
            
            # Separation of concerns
            separation_score = self._evaluate_separation_of_concerns(refactored_code)
            
            maintainability_score = (
                0.25 * method_length_score +
                0.2 * cohesion_score +
                0.2 * (1.0 - coupling_score) +  # Lower coupling is better
                0.2 * (1.0 - duplication_score) +  # Lower duplication is better
                0.15 * separation_score
            )
            
            return min(max(maintainability_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating maintainability: {e}")
            return 0.0
    
    def _apply_refactoring(self, individual: Individual, code_analyzer) -> str:
        """Apply refactoring based on individual's genes."""
        # This would interface with the actual refactoring engine
        # For now, return a mock refactored code
        
        genes = individual.genes
        original_code = code_analyzer.original_code
        
        # Extract refactoring parameters from genes
        extract_method_threshold = genes[0]
        complexity_weight = genes[1]
        readability_weight = genes[2]
        performance_weight = genes[3]
        maintainability_weight = genes[4]
        confidence_threshold = genes[5]
        similarity_threshold = genes[6]
        aggressiveness = genes[7]
        
        # Apply refactoring transformations based on genes
        # This is a simplified simulation
        refactored_code = self._simulate_refactoring(
            original_code,
            extract_method_threshold,
            aggressiveness
        )
        
        individual.refactored_code = refactored_code
        return refactored_code
    
    def _simulate_refactoring(self, code: str, threshold: float, aggressiveness: float) -> str:
        """Simulate refactoring transformations."""
        # Simple simulation of refactoring
        lines = code.split('\n')
        refactored_lines = []
        
        for line in lines:
            # Simulate extract method refactoring
            if threshold > 0.7 and len(line.strip()) > 80:
                # Break long lines
                if '=' in line and '+' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        expression = parts[1].strip()
                        refactored_lines.append(f"    # Extracted calculation")
                        refactored_lines.append(f"    temp_result = {expression}")
                        refactored_lines.append(f"    {var_name} = temp_result")
                        continue
            
            # Simulate variable extraction
            if aggressiveness > 0.6 and 'if' in line and '>' in line:
                # Extract magic numbers
                import re
                numbers = re.findall(r'\b\d+\b', line)
                for num in numbers:
                    if int(num) > 10:  # Consider as magic number
                        const_name = f"THRESHOLD_{num}"
                        line = line.replace(num, const_name)
                        refactored_lines.append(f"    {const_name} = {num}")
                        break
            
            refactored_lines.append(line)
        
        return '\n'.join(refactored_lines)
    
    def _calculate_quality_metrics(self, code: str) -> Dict[str, float]:
        """Calculate various quality metrics for code."""
        metrics = {}
        
        try:
            # Radon metrics
            raw_metrics = analyze(code)
            
            # Lines of code
            loc = raw_metrics.loc
            sloc = raw_metrics.sloc
            
            # Maintainability index (simplified)
            complexity = self._get_cyclomatic_complexity(code)
            if loc > 0:
                mi = max(0, (171 - 5.2 * np.log(loc) - 0.23 * complexity - 16.2 * np.log(loc)) / 171)
            else:
                mi = 0.0
            
            metrics["maintainability_index"] = mi
            metrics["complexity_normalized"] = min(complexity / 10.0, 1.0)  # Normalize to 0-1
            metrics["readability_score"] = self._calculate_readability_score(code)
            metrics["documentation_ratio"] = self._calculate_documentation_ratio(code)
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            metrics = {
                "maintainability_index": 0.5,
                "complexity_normalized": 0.5,
                "readability_score": 0.5,
                "documentation_ratio": 0.0
            }
        
        return metrics
    
    def _get_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        try:
            from radon.complexity import cc_visit
            complexity_list = cc_visit(code)
            total_complexity = sum(item.complexity for item in complexity_list)
            return total_complexity
        except:
            return 1
    
    def _calculate_readability_score(self, code: str) -> float:
        """Calculate readability score based on various factors."""
        score = 0.0
        lines = code.split('\n')
        
        # Average line length
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
            # Ideal line length is around 80 characters
            line_length_score = max(0, 1.0 - abs(avg_line_length - 80) / 80)
        else:
            line_length_score = 0.0
        
        # Variable naming (simple heuristic)
        naming_score = self._evaluate_variable_naming(code)
        
        # Indentation consistency
        indentation_score = self._evaluate_indentation(code)
        
        score = (line_length_score + naming_score + indentation_score) / 3.0
        return min(max(score, 0.0), 1.0)
    
    def _calculate_documentation_ratio(self, code: str) -> float:
        """Calculate ratio of commented lines to total lines."""
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return 0.0
        
        return min(comment_lines / total_lines, 1.0)
    
    def _evaluate_line_length(self, code: str) -> float:
        """Evaluate line length appropriateness."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 1.0
        
        # Penalty for lines that are too long or too short
        ideal_length = 80
        total_penalty = 0.0
        
        for line in non_empty_lines:
            length = len(line)
            if length > 120:  # Too long
                total_penalty += (length - 120) / 120
            elif length < 10:  # Too short (excluding empty lines)
                total_penalty += (10 - length) / 10
        
        avg_penalty = total_penalty / len(non_empty_lines)
        return max(0.0, 1.0 - avg_penalty)
    
    def _evaluate_naming_conventions(self, tree: ast.AST) -> float:
        """Evaluate naming conventions."""
        score = 0.0
        total_names = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_names += 1
                # Check if function name is snake_case
                if node.name.islower() and '_' in node.name:
                    score += 1
                elif node.name.islower():
                    score += 0.7
            
            elif isinstance(node, ast.ClassDef):
                total_names += 1
                # Check if class name is PascalCase
                if node.name[0].isupper() and node.name.isalnum():
                    score += 1
                else:
                    score += 0.5
            
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):  # Variable assignment
                    total_names += 1
                    # Check if variable name is descriptive
                    if len(node.id) > 2 and node.id.islower():
                        score += 1
                    elif len(node.id) > 1:
                        score += 0.5
        
        return score / max(total_names, 1)
    
    def _evaluate_method_complexity(self, code: str) -> float:
        """Evaluate method complexity."""
        complexity = self._get_cyclomatic_complexity(code)
        # Normalize complexity score (lower complexity is better)
        return max(0.0, 1.0 - complexity / 20.0)
    
    def _evaluate_code_structure(self, tree: ast.AST) -> float:
        """Evaluate code structure quality."""
        score = 0.0
        
        # Count nested levels
        max_nesting = self._get_max_nesting_level(tree)
        nesting_score = max(0.0, 1.0 - max_nesting / 10.0)
        
        # Count functions and classes
        functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        
        # Prefer modular code
        if functions > 0:
            score += 0.5
        if classes > 0:
            score += 0.3
        
        score = (score + nesting_score) / 2.0
        return min(max(score, 0.0), 1.0)
    
    def _evaluate_comments(self, code: str) -> float:
        """Evaluate comment quality."""
        lines = code.split('\n')
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if not code_lines:
            return 0.0
        
        # Ideal comment ratio is around 10-20%
        comment_ratio = len(comment_lines) / len(code_lines)
        
        if 0.1 <= comment_ratio <= 0.2:
            return 1.0
        elif comment_ratio < 0.1:
            return comment_ratio / 0.1
        else:
            return max(0.0, 1.0 - (comment_ratio - 0.2) / 0.3)
    
    def _count_nested_loops(self, code: str) -> int:
        """Count nested loops in code."""
        try:
            tree = ast.parse(code)
            max_nesting = 0
            
            def count_nesting(node, current_depth=0):
                nonlocal max_nesting
                
                if isinstance(node, (ast.For, ast.While)):
                    current_depth += 1
                    max_nesting = max(max_nesting, current_depth)
                
                for child in ast.iter_child_nodes(node):
                    count_nesting(child, current_depth)
            
            count_nesting(tree)
            return max_nesting
        except:
            return 0
    
    def _evaluate_method_calls(self, code: str) -> float:
        """Evaluate method call efficiency."""
        try:
            tree = ast.parse(code)
            method_calls = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))
            lines = len(code.strip().split('\n'))
            
            if lines == 0:
                return 0.0
            
            # Ratio of method calls to lines of code
            call_ratio = method_calls / lines
            
            # Optimal ratio is around 0.2-0.5
            if 0.2 <= call_ratio <= 0.5:
                return 1.0
            elif call_ratio < 0.2:
                return call_ratio / 0.2
            else:
                return max(0.0, 1.0 - (call_ratio - 0.5) / 0.5)
        except:
            return 0.5
    
    def _evaluate_method_length(self, code: str) -> float:
        """Evaluate method length appropriateness."""
        try:
            tree = ast.parse(code)
            method_lengths = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count lines in method
                    method_lines = node.end_lineno - node.lineno + 1
                    method_lengths.append(method_lines)
            
            if not method_lengths:
                return 1.0
            
            # Ideal method length is 10-30 lines
            avg_length = sum(method_lengths) / len(method_lengths)
            
            if 10 <= avg_length <= 30:
                return 1.0
            elif avg_length < 10:
                return avg_length / 10
            else:
                return max(0.0, 1.0 - (avg_length - 30) / 50)
        except:
            return 0.5
    
    def _evaluate_class_cohesion(self, code: str) -> float:
        """Evaluate class cohesion."""
        # Simplified cohesion metric
        try:
            tree = ast.parse(code)
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            if not classes:
                return 1.0
            
            total_cohesion = 0.0
            
            for class_node in classes:
                methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
                attributes = set()
                
                # Find class attributes
                for node in ast.walk(class_node):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                        attributes.add(node.attr)
                
                if not methods or not attributes:
                    cohesion = 1.0
                else:
                    # Calculate how many methods use how many attributes
                    method_attribute_usage = 0
                    for method in methods:
                        used_attributes = set()
                        for node in ast.walk(method):
                            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                                if node.attr in attributes:
                                    used_attributes.add(node.attr)
                        method_attribute_usage += len(used_attributes)
                    
                    max_possible = len(methods) * len(attributes)
                    cohesion = method_attribute_usage / max(max_possible, 1)
                
                total_cohesion += cohesion
            
            return total_cohesion / len(classes)
        except:
            return 0.5
    
    def _evaluate_coupling(self, code: str) -> float:
        """Evaluate coupling between classes/modules."""
        try:
            tree = ast.parse(code)
            imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            
            if classes == 0:
                return 0.0
            
            # Simple coupling metric based on imports per class
            coupling_ratio = imports / classes
            
            # Lower coupling is better, normalize to 0-1 scale
            return min(coupling_ratio / 5.0, 1.0)
        except:
            return 0.5
    
    def _evaluate_code_duplication(self, code: str) -> float:
        """Evaluate code duplication."""
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return 0.0
        
        # Simple duplication detection
        unique_lines = set(lines)
        duplication_ratio = 1.0 - (len(unique_lines) / len(lines))
        
        return duplication_ratio
    
    def _evaluate_separation_of_concerns(self, code: str) -> float:
        """Evaluate separation of concerns."""
        try:
            tree = ast.parse(code)
            
            # Count different types of operations in functions
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                return 1.0
            
            total_separation = 0.0
            
            for func in functions:
                # Count different types of operations
                io_operations = 0
                calculations = 0
                control_flow = 0
                
                for node in ast.walk(func):
                    if isinstance(node, ast.Call):
                        func_name = getattr(node.func, 'id', '') or getattr(node.func, 'attr', '')
                        if func_name in ['print', 'input', 'open', 'read', 'write']:
                            io_operations += 1
                        else:
                            calculations += 1
                    elif isinstance(node, (ast.If, ast.For, ast.While)):
                        control_flow += 1
                
                # A function should ideally focus on one type of operation
                total_operations = io_operations + calculations + control_flow
                if total_operations == 0:
                    separation = 1.0
                else:
                    max_type = max(io_operations, calculations, control_flow)
                    separation = max_type / total_operations
                
                total_separation += separation
            
            return total_separation / len(functions)
        except:
            return 0.5
    
    def _evaluate_variable_naming(self, code: str) -> float:
        """Evaluate variable naming quality."""
        try:
            tree = ast.parse(code)
            total_score = 0.0
            total_variables = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    total_variables += 1
                    var_name = node.id
                    
                    # Score based on naming conventions
                    score = 0.0
                    
                    # Length (not too short, not too long)
                    if 3 <= len(var_name) <= 20:
                        score += 0.3
                    elif len(var_name) > 1:
                        score += 0.1
                    
                    # No single letter variables (except for loop counters)
                    if len(var_name) > 1 or var_name in ['i', 'j', 'k']:
                        score += 0.2
                    
                    # Snake case
                    if var_name.islower() and ('_' in var_name or var_name.isalpha()):
                        score += 0.3
                    
                    # Descriptive (contains vowels, not all consonants)
                    vowels = set('aeiou')
                    if any(c in vowels for c in var_name.lower()):
                        score += 0.2
                    
                    total_score += score
            
            return total_score / max(total_variables, 1)
        except:
            return 0.5
    
    def _evaluate_indentation(self, code: str) -> float:
        """Evaluate indentation consistency."""
        lines = code.split('\n')
        indentations = []
        
        for line in lines:
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if not indentations:
            return 1.0
        
        # Check for consistent indentation (should be multiples of 4 or 2)
        consistent_4 = all(indent % 4 == 0 for indent in indentations)
        consistent_2 = all(indent % 2 == 0 for indent in indentations)
        
        if consistent_4:
            return 1.0
        elif consistent_2:
            return 0.8
        else:
            return 0.3
    
    def _get_max_nesting_level(self, tree: ast.AST) -> int:
        """Get maximum nesting level in AST."""
        max_depth = 0
        
        def traverse(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            # Increase depth for control structures
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                depth += 1
            
            for child in ast.iter_child_nodes(node):
                traverse(child, depth)
        
        traverse(tree)
        return max_depth


def create_objective_functions(config: Dict) -> List:
    """Create objective functions for NSGA-II."""
    evaluator = FitnessEvaluator(config)
    
    objective_functions = []
    
    for objective in config.get("genetic_algorithm", {}).get("objectives", []):
        obj_name = objective["name"]
        
        if obj_name == "code_quality":
            objective_functions.append(evaluator.evaluate_code_quality)
        elif obj_name == "readability":
            objective_functions.append(evaluator.evaluate_readability)
        elif obj_name == "performance":
            objective_functions.append(evaluator.evaluate_performance)
        elif obj_name == "maintainability":
            objective_functions.append(evaluator.evaluate_maintainability)
        else:
            logger.warning(f"Unknown objective: {obj_name}")
    
    return objective_functions


class CodeAnalyzer:
    """Analyzes code for refactoring optimization."""
    
    def __init__(self, original_code: str):
        self.original_code = original_code
        self.ast_tree = None
        self.metrics = {}
        
        try:
            self.ast_tree = ast.parse(original_code)
            self._calculate_baseline_metrics()
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
    
    def _calculate_baseline_metrics(self):
        """Calculate baseline metrics for the original code."""
        if self.ast_tree is None:
            return
        
        evaluator = FitnessEvaluator({})
        
        self.metrics = {
            "original_complexity": evaluator._get_cyclomatic_complexity(self.original_code),
            "original_lines": len(self.original_code.strip().split('\n')),
            "original_methods": len([n for n in ast.walk(self.ast_tree) if isinstance(n, ast.FunctionDef)]),
            "original_classes": len([n for n in ast.walk(self.ast_tree) if isinstance(n, ast.ClassDef)]),
            "original_nesting": evaluator._get_max_nesting_level(self.ast_tree)
        }
    
    def get_refactoring_candidates(self) -> List[Dict]:
        """Identify potential refactoring candidates."""
        candidates = []
        
        if self.ast_tree is None:
            return candidates
        
        # Find long methods
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                method_lines = node.end_lineno - node.lineno + 1
                if method_lines > 20:
                    candidates.append({
                        "type": "extract_method",
                        "location": f"line {node.lineno}",
                        "reason": f"Method '{node.name}' is {method_lines} lines long",
                        "priority": min(method_lines / 20.0, 3.0)
                    })
        
        # Find complex conditionals
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.If):
                # Count nested conditions
                nested_ifs = len([n for n in ast.walk(node) if isinstance(n, ast.If)])
                if nested_ifs > 3:
                    candidates.append({
                        "type": "simplify_conditional",
                        "location": f"line {node.lineno}",
                        "reason": f"Complex conditional with {nested_ifs} nested conditions",
                        "priority": min(nested_ifs / 3.0, 3.0)
                    })
        
        # Find magic numbers
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Num) and isinstance(node.n, (int, float)):
                if node.n > 10 and node.n != 100:  # Common thresholds
                    candidates.append({
                        "type": "replace_magic_numbers",
                        "location": f"line {node.lineno}",
                        "reason": f"Magic number {node.n} found",
                        "priority": 1.5
                    })
        
        # Sort by priority
        candidates.sort(key=lambda x: x["priority"], reverse=True)
        return candidates
    
    def estimate_refactoring_impact(self, refactoring_type: str) -> Dict[str, float]:
        """Estimate the impact of a refactoring type."""
        impact = {
            "code_quality": 0.0,
            "readability": 0.0,
            "performance": 0.0,
            "maintainability": 0.0
        }
        
        # Predefined impact estimates for different refactoring types
        refactoring_impacts = {
            "extract_method": {
                "code_quality": 0.6,
                "readability": 0.7,
                "performance": 0.3,
                "maintainability": 0.8
            },
            "extract_variable": {
                "code_quality": 0.4,
                "readability": 0.8,
                "performance": 0.1,
                "maintainability": 0.5
            },
            "replace_magic_numbers": {
                "code_quality": 0.5,
                "readability": 0.6,
                "performance": 0.0,
                "maintainability": 0.7
            },
            "simplify_conditional": {
                "code_quality": 0.7,
                "readability": 0.8,
                "performance": 0.4,
                "maintainability": 0.6
            },
            "remove_dead_code": {
                "code_quality": 0.8,
                "readability": 0.5,
                "performance": 0.6,
                "maintainability": 0.7
            }
        }
        
        return refactoring_impacts.get(refactoring_type, impact)