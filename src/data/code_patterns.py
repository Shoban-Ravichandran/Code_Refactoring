"""
Code pattern generators for different refactoring types.
"""

import random
import ast
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class CodePattern:
    """Represents a code pattern with before and after refactoring."""
    original_code: str
    refactored_code: str
    pattern_type: str
    description: str
    quality_improvement: Dict[str, float]
    complexity_reduction: float
    difficulty_level: str = "medium"
    tags: List[str] = None

class PatternGenerator(ABC):
    """Abstract base class for pattern generators."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.variable_names = self._load_variable_names()
        self.method_names = self._load_method_names()
        self.class_names = self._load_class_names()
    
    def _load_variable_names(self) -> List[str]:
        """Load common variable names."""
        return [
            "data", "result", "value", "item", "element", "node", "temp", "buffer",
            "index", "count", "size", "length", "total", "sum", "average", "max",
            "min", "current", "previous", "next", "first", "last", "input", "output",
            "source", "target", "config", "settings", "options", "params", "args",
            "response", "request", "message", "error", "status", "flag", "state"
        ]
    
    def _load_method_names(self) -> List[str]:
        """Load common method names."""
        return [
            "calculate", "process", "validate", "transform", "convert", "parse",
            "format", "serialize", "deserialize", "execute", "run", "start", "stop",
            "initialize", "finalize", "create", "destroy", "build", "update", "delete",
            "find", "search", "filter", "sort", "compare", "merge", "split", "join",
            "encode", "decode", "encrypt", "decrypt", "compress", "decompress",
            "handle", "manage", "control", "monitor", "check", "verify", "test"
        ]
    
    def _load_class_names(self) -> List[str]:
        """Load common class names."""
        return [
            "Manager", "Handler", "Controller", "Service", "Repository", "Factory",
            "Builder", "Validator", "Transformer", "Processor", "Analyzer", "Parser",
            "Formatter", "Serializer", "Deserializer", "Executor", "Runner", "Worker",
            "Client", "Server", "Provider", "Consumer", "Producer", "Observer",
            "Listener", "Strategy", "Command", "State", "Context", "Adapter",
            "Decorator", "Facade", "Proxy", "Singleton", "Registry", "Cache"
        ]
    
    @abstractmethod
    def generate_pattern(self, complexity: str = "medium") -> CodePattern:
        """Generate a code pattern."""
        pass
    
    def generate_multiple_patterns(self, count: int, complexity: str = "medium") -> List[CodePattern]:
        """Generate multiple patterns."""
        return [self.generate_pattern(complexity) for _ in range(count)]

class ExtractMethodGenerator(PatternGenerator):
    """Generate extract method refactoring patterns."""
    
    def generate_pattern(self, complexity: str = "medium") -> CodePattern:
        """Generate extract method pattern."""
        method_name = random.choice(self.method_names)
        var_names = random.sample(self.variable_names, k=random.randint(4, 8))
        
        # Determine complexity level
        if complexity == "low":
            lines_to_extract = random.randint(3, 5)
            conditions = random.randint(1, 2)
        elif complexity == "high":
            lines_to_extract = random.randint(8, 15)
            conditions = random.randint(4, 7)
        else:  # medium
            lines_to_extract = random.randint(5, 8)
            conditions = random.randint(2, 4)
        
        # Generate original code with long method
        original_lines = [
            f"def {method_name}(self, {', '.join(var_names[:2])}):",
            f"    # Long method that needs refactoring",
            f"    {var_names[2]} = {var_names[0]} + {var_names[1]}"
        ]
        
        # Add complex logic
        for i in range(lines_to_extract):
            if i < conditions:
                threshold = random.randint(10, 200)
                operation = random.choice(["+", "-", "*", "/"])
                factor = random.uniform(1.1, 3.0)
                original_lines.extend([
                    f"    if {var_names[2]} > {threshold}:",
                    f"        {var_names[3]} = {var_names[2]} {operation} {factor:.1f}",
                    f"        {var_names[4]} = {var_names[3]} / {random.randint(2, 5)}",
                    f"        if {var_names[4]} > {threshold//2}:",
                    f"            {var_names[5]} = {var_names[4]} - {random.randint(5, 15)}",
                    f"            return {var_names[5]}",
                    f"        else:",
                    f"            {var_names[5]} = {var_names[4]} + {random.randint(10, 30)}",
                    f"            return {var_names[5]}"
                ])
            else:
                calc_var = var_names[min(len(var_names)-1, 3+i)]
                original_lines.append(f"    {calc_var} = {var_names[2]} * {random.uniform(1.5, 4.0):.1f}")
        
        original_lines.append(f"    return {var_names[3] if len(var_names) > 3 else var_names[2]}")
        original_code = "\n".join(original_lines)
        
        # Generate refactored code
        extracted_method = f"_calculate_{random.choice(['result', 'value', 'total', 'score'])}"
        refactored_lines = [
            f"def {method_name}(self, {', '.join(var_names[:2])}):",
            f"    {var_names[2]} = {var_names[0]} + {var_names[1]}",
            f"    return self.{extracted_method}({var_names[2]})",
            f"",
            f"def {extracted_method}(self, {var_names[2]}):",
            f"    \"\"\"Extract complex calculation logic.\"\"\""
        ]
        
        # Add extracted logic
        for i in range(conditions):
            threshold = random.randint(10, 200)
            operation = random.choice(["+", "-", "*"])
            factor = random.uniform(1.1, 3.0)
            refactored_lines.extend([
                f"    if {var_names[2]} > {threshold}:",
                f"        {var_names[3]} = {var_names[2]} {operation} {factor:.1f}",
                f"        {var_names[4]} = {var_names[3]} / {random.randint(2, 5)}",
                f"        return {var_names[4]} - {random.randint(5, 15)} if {var_names[4]} > {threshold//2} else {var_names[4]} + {random.randint(10, 30)}"
            ])
        
        refactored_lines.append(f"    return {var_names[2]} * {random.uniform(1.5, 4.0):.1f}")
        refactored_code = "\n".join(refactored_lines)
        
        quality_improvement = {
            "readability": random.uniform(0.6, 0.9),
            "maintainability": random.uniform(0.5, 0.8),
            "reusability": random.uniform(0.4, 0.7),
            "testability": random.uniform(0.5, 0.8)
        }
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="extract_method",
            description="Extract complex logic into separate method",
            quality_improvement=quality_improvement,
            complexity_reduction=random.uniform(0.3, 0.7),
            difficulty_level=complexity,
            tags=["method", "extraction", "complexity"]
        )

class ExtractVariableGenerator(PatternGenerator):
    """Generate extract variable refactoring patterns."""
    
    def generate_pattern(self, complexity: str = "medium") -> CodePattern:
        """Generate extract variable pattern."""
        var_names = random.sample(self.variable_names, k=random.randint(5, 9))
        method_name = f"calculate_{random.choice(['price', 'total', 'score', 'rate', 'value'])}"
        
        # Generate complex expressions based on complexity
        if complexity == "low":
            expressions = 1
            operators = ["+", "*"]
        elif complexity == "high":
            expressions = 3
            operators = ["+", "-", "*", "/", "**"]
        else:  # medium
            expressions = 2
            operators = ["+", "-", "*", "/"]
        
        # Generate original code with complex expressions
        original_lines = [f"def {method_name}(self, {', '.join(var_names[:3])}):"
        ]
        
        # Create complex expression
        expr_parts = []
        for i in range(expressions):
            factor1 = random.uniform(0.5, 2.5)
            factor2 = random.uniform(0.5, 2.5)
            op = random.choice(operators)
            expr_parts.append(f"({var_names[i]} * {factor1:.1f} {op} {var_names[i+1]} * {factor2:.1f})")
        
        complex_expr = " + ".join(expr_parts)
        
        # Add conditional multiplier
        threshold = random.randint(50, 200)
        multiplier1 = random.uniform(1.1, 1.5)
        multiplier2 = random.uniform(1.0, 1.2)
        
        original_code = f"""def {method_name}(self, {', '.join(var_names[:3])}):
    return {complex_expr} * ({multiplier1:.2f} if {var_names[0]} > {threshold} else {multiplier2:.2f})"""
        
        # Generate refactored code
        base_var = f"{var_names[3]}_base"
        multiplier_var = f"{var_names[4]}_multiplier"
        
        refactored_lines = [
            f"def {method_name}(self, {', '.join(var_names[:3])}):",
            f"    {base_var} = {complex_expr}",
            f"    {multiplier_var} = {multiplier1:.2f} if {var_names[0]} > {threshold} else {multiplier2:.2f}",
            f"    return {base_var} * {multiplier_var}"
        ]
        
        refactored_code = "\n".join(refactored_lines)
        
        quality_improvement = {
            "readability": random.uniform(0.7, 0.95),
            "maintainability": random.uniform(0.5, 0.8),
            "debuggability": random.uniform(0.6, 0.9),
            "testability": random.uniform(0.4, 0.7)
        }
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="extract_variable",
            description="Extract complex expressions into named variables",
            quality_improvement=quality_improvement,
            complexity_reduction=random.uniform(0.2, 0.5),
            difficulty_level=complexity,
            tags=["variable", "extraction", "expression"]
        )

class ReplaceMagicNumbersGenerator(PatternGenerator):
    """Generate replace magic numbers refactoring patterns."""
    
    def generate_pattern(self, complexity: str = "medium") -> CodePattern:
        """Generate replace magic numbers pattern."""
        var_names = random.sample(self.variable_names, k=4)
        method_name = f"process_{random.choice(['data', 'request', 'input', 'payment', 'order'])}"
        
        # Define magic numbers based on complexity
        if complexity == "low":
            magic_numbers = [random.randint(5, 100)]
        elif complexity == "high":
            magic_numbers = [random.randint(5, 50), random.uniform(0.1, 0.9), random.randint(100, 1000), 3.14159]
        else:  # medium
            magic_numbers = [random.randint(10, 200), random.uniform(0.1, 0.5)]
        
        # Generate original code with magic numbers
        original_lines = [
            f"def {method_name}(self, {var_names[0]}):"
        ]
        
        for i, magic_num in enumerate(magic_numbers):
            var_name = var_names[min(i + 1, len(var_names) - 1)]
            if isinstance(magic_num, float):
                original_lines.extend([
                    f"    if {var_names[0]} > {magic_numbers[0] if i > 0 else 50}:",
                    f"        {var_name} = {var_names[0]} * {magic_num:.3f}",
                    f"        return {var_name}"
                ])
            else:
                original_lines.extend([
                    f"    if {var_names[0]} > {magic_num}:",
                    f"        {var_name} = {var_names[0]} * {random.uniform(1.1, 2.0):.1f}",
                    f"        return {var_name}"
                ])
        
        original_lines.append(f"    return {var_names[0]}")
        original_code = "\n".join(original_lines)
        
        # Generate refactored code with constants
        const_names = []
        for i, magic_num in enumerate(magic_numbers):
            if isinstance(magic_num, float):
                const_name = f"{random.choice(['RATE', 'FACTOR', 'MULTIPLIER'])}_{random.choice(['VALUE', 'CONSTANT'])}"
            else:
                const_name = f"{random.choice(['MAX', 'MIN', 'DEFAULT'])}_{random.choice(['THRESHOLD', 'LIMIT', 'VALUE'])}"
            const_names.append(const_name)
        
        refactored_lines = ["# Constants"]
        for const_name, magic_num in zip(const_names, magic_numbers):
            if isinstance(magic_num, float):
                refactored_lines.append(f"{const_name} = {magic_num:.3f}")
            else:
                refactored_lines.append(f"{const_name} = {magic_num}")
        
        refactored_lines.extend([
            "",
            f"def {method_name}(self, {var_names[0]}):"
        ])
        
        for i, const_name in enumerate(const_names):
            var_name = var_names[min(i + 1, len(var_names) - 1)]
            if i == 0:
                refactored_lines.extend([
                    f"    if {var_names[0]} > {const_name}:",
                    f"        {var_name} = {var_names[0]} * {const_names[1] if len(const_names) > 1 else random.uniform(1.1, 2.0):.1f}",
                    f"        return {var_name}"
                ])
            elif len(const_names) > 1:
                refactored_lines.extend([
                    f"    elif {var_names[0]} > {const_names[0]}:",
                    f"        {var_name} = {var_names[0]} * {const_name}",
                    f"        return {var_name}"
                ])
        
        refactored_lines.append(f"    return {var_names[0]}")
        refactored_code = "\n".join(refactored_lines)
        
        quality_improvement = {
            "readability": random.uniform(0.6, 0.8),
            "maintainability": random.uniform(0.7, 0.9),
            "configuration": random.uniform(0.8, 0.95),
            "documentation": random.uniform(0.5, 0.8)
        }
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="replace_magic_numbers",
            description="Replace magic numbers with named constants",
            quality_improvement=quality_improvement,
            complexity_reduction=random.uniform(0.1, 0.4),
            difficulty_level=complexity,
            tags=["constants", "magic_numbers", "configuration"]
        )

class SimplifyConditionalGenerator(PatternGenerator):
    """Generate simplify conditional refactoring patterns."""
    
    def generate_pattern(self, complexity: str = "medium") -> CodePattern:
        """Generate simplify conditional pattern."""
        var_names = random.sample(self.variable_names, k=4)
        method_name = f"check_{random.choice(['status', 'condition', 'state', 'validity', 'eligibility'])}"
        
        # Determine nesting level based on complexity
        if complexity == "low":
            nesting_levels = 2
            conditions = 2
        elif complexity == "high":
            nesting_levels = 4
            conditions = 5
        else:  # medium
            nesting_levels = 3
            conditions = 3
        
        # Generate original code with nested conditionals
        original_lines = [
            f"def {method_name}(self, {', '.join(var_names[:2])}):"
        ]
        
        # Create nested structure
        indent = "    "
        for i in range(nesting_levels):
            if i == 0:
                condition = f"{var_names[0]} > 0"
            elif i == 1:
                condition = f"{var_names[1]} is not None"
            elif i == 2:
                condition = f'{var_names[1]} != ""'
            else:
                condition = f"len({var_names[1]}) > {random.randint(1, 10)}"
            
            original_lines.append(f"{indent}if {condition}:")
            indent += "    "
        
        # Add return statements
        original_lines.append(f"{indent}return True")
        
        # Add else clauses
        for i in range(nesting_levels):
            indent = indent[:-4]
            original_lines.append(f"{indent}else:")
            original_lines.append(f"{indent}    return False")
        
        original_code = "\n".join(original_lines)
        
        # Generate refactored code with simplified logic
        conditions_list = [
            f"{var_names[0]} > 0",
            f"{var_names[1]} is not None",
            f'{var_names[1]} != ""'
        ]
        
        if nesting_levels > 3:
            conditions_list.append(f"len({var_names[1]}) > {random.randint(1, 10)}")
        
        simplified_condition = " and ".join(conditions_list[:nesting_levels])
        
        refactored_code = f"""def {method_name}(self, {', '.join(var_names[:2])}):
    return {simplified_condition}"""
        
        quality_improvement = {
            "readability": random.uniform(0.7, 0.9),
            "maintainability": random.uniform(0.6, 0.8),
            "conciseness": random.uniform(0.8, 0.95),
            "complexity": random.uniform(0.5, 0.8)
        }
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="simplify_conditional",
            description="Simplify nested conditional statements",
            quality_improvement=quality_improvement,
            complexity_reduction=random.uniform(0.4, 0.8),
            difficulty_level=complexity,
            tags=["conditional", "simplification", "logic"]
        )

class RemoveDeadCodeGenerator(PatternGenerator):
    """Generate remove dead code refactoring patterns."""
    
    def generate_pattern(self, complexity: str = "medium") -> CodePattern:
        """Generate remove dead code pattern."""
        var_names = random.sample(self.variable_names, k=5)
        method_name = f"process_{random.choice(['request', 'data', 'input', 'transaction'])}"
        
        # Generate original code with dead code
        original_lines = [
            f"def {method_name}(self, {var_names[0]}):",
            f"    {var_names[1]} = {var_names[0]} * 2",
            f"    "
        ]
        
        # Add dead code based on complexity
        if complexity == "low":
            original_lines.extend([
                f"    # Dead code - never executed",
                f"    if False:",
                f"        {var_names[2]} = {var_names[1]} + 100",
                f"        return {var_names[2]}"
            ])
        elif complexity == "high":
            original_lines.extend([
                f"    # Dead code - unreachable",
                f"    if False:",
                f"        {var_names[2]} = {var_names[1]} + 100",
                f"        print('This will never execute')",
                f"        for i in range(10):",
                f"            {var_names[2]} += i",
                f"        return {var_names[2]}",
                f"    ",
                f"    # Unused variables",
                f"    {var_names[3]} = 'unused_string'",
                f"    {var_names[4]} = [1, 2, 3, 4, 5]",
                f"    unused_dict = {{'key': 'value'}}",
                f"    ",
                f"    # More dead code",
                f"    if 1 == 2:",
                f"        print('Impossible condition')"
            ])
        else:  # medium
            original_lines.extend([
                f"    # Dead code - never executed",
                f"    if False:",
                f"        {var_names[2]} = {var_names[1]} + 100",
                f"        print('This will never execute')",
                f"        return {var_names[2]}",
                f"    ",
                f"    # Unused variable",
                f"    {var_names[3]} = 'unused'"
            ])
        
        original_lines.extend([
            f"    ",
            f"    return {var_names[1]}"
        ])
        
        original_code = "\n".join(original_lines)
        
        # Generate refactored code without dead code
        refactored_code = f"""def {method_name}(self, {var_names[0]}):
    {var_names[1]} = {var_names[0]} * 2
    return {var_names[1]}"""
        
        quality_improvement = {
            "readability": random.uniform(0.5, 0.7),
            "maintainability": random.uniform(0.7, 0.9),
            "performance": random.uniform(0.2, 0.5),
            "size": random.uniform(0.6, 0.9)
        }
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="remove_dead_code",
            description="Remove unreachable and unused code",
            quality_improvement=quality_improvement,
            complexity_reduction=random.uniform(0.3, 0.7),
            difficulty_level=complexity,
            tags=["dead_code", "cleanup", "optimization"]
        )

class ExtractClassGenerator(PatternGenerator):
    """Generate extract class refactoring patterns."""
    
    def generate_pattern(self, complexity: str = "medium") -> CodePattern:
        """Generate extract class pattern."""
        class_name = random.choice(self.class_names)
        method_names = random.sample(self.method_names, k=4)
        var_names = random.sample(self.variable_names, k=6)
        
        # Determine class complexity
        if complexity == "low":
            methods_to_extract = 1
            attributes_count = 2
        elif complexity == "high":
            methods_to_extract = 3
            attributes_count = 6
        else:  # medium
            methods_to_extract = 2
            attributes_count = 4
        
        # Generate original god class
        original_lines = [
            f"class {class_name}:",
            f"    def __init__(self):"
        ]
        
        # Add many attributes
        for i in range(attributes_count):
            if i < 2:
                original_lines.append(f"        self.{var_names[i]} = []")
            elif i < 4:
                original_lines.append(f"        self.{var_names[i]} = {{}}")
            else:
                original_lines.append(f"        self.{var_names[i]} = {random.randint(0, 100)}")
        
        original_lines.append("")
        
        # Add core methods
        for i in range(2):
            original_lines.extend([
                f"    def {method_names[i]}(self, item):",
                f"        self.{var_names[0]}.append(item)",
                f"        self.{var_names[4]} += 1",
                ""
            ])
        
        # Add methods that should be extracted
        for i in range(methods_to_extract):
            method_idx = i + 2
            original_lines.extend([
                f"    def {method_names[method_idx]}(self, message):",
                f"        self.{var_names[5]} = message",
                f"        print(f'Log: {{message}}')",
                f"        # Additional logging logic here",
                ""
            ])
        
        original_code = "\n".join(original_lines[:-1])  # Remove last empty line
        
        # Generate refactored code with extracted class
        extracted_class = f"{random.choice(['Logger', 'Handler', 'Tracker', 'Monitor'])}"
        
        refactored_lines = [
            f"class {extracted_class}:",
            f"    def __init__(self):",
            f"        self.{var_names[5]} = ''",
            ""
        ]
        
        # Add extracted methods
        for i in range(methods_to_extract):
            method_idx = i + 2
            refactored_lines.extend([
                f"    def {method_names[method_idx]}(self, message):",
                f"        self.{var_names[5]} = message",
                f"        print(f'Log: {{message}}')",
                f"        # Additional logging logic here",
                ""
            ])
        
        refactored_lines.extend([
            f"class {class_name}:",
            f"    def __init__(self):"
        ])
        
        # Add remaining attributes
        for i in range(attributes_count - methods_to_extract):
            if i < 2:
                refactored_lines.append(f"        self.{var_names[i]} = []")
            elif i < 4:
                refactored_lines.append(f"        self.{var_names[i]} = {{}}")
            else:
                refactored_lines.append(f"        self.{var_names[i]} = {random.randint(0, 100)}")
        
        refactored_lines.append(f"        self.{extracted_class.lower()} = {extracted_class}()")
        refactored_lines.append("")
        
        # Add core methods
        for i in range(2):
            refactored_lines.extend([
                f"    def {method_names[i]}(self, item):",
                f"        self.{var_names[0]}.append(item)",
                f"        self.{var_names[4 if attributes_count > 4 else 2]} += 1",
                ""
            ])
        
        refactored_code = "\n".join(refactored_lines[:-1])  # Remove last empty line
        
        quality_improvement = {
            "separation_of_concerns": random.uniform(0.7, 0.9),
            "maintainability": random.uniform(0.6, 0.8),
            "reusability": random.uniform(0.8, 0.95),
            "testability": random.uniform(0.5, 0.8)
        }
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="extract_class",
            description="Extract related functionality into separate class",
            quality_improvement=quality_improvement,
            complexity_reduction=random.uniform(0.4, 0.7),
            difficulty_level=complexity,
            tags=["class", "extraction", "separation"]
        )

class CodePatternGenerator:
    """Main class for generating various code patterns."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.generators = {
            "extract_method": ExtractMethodGenerator(config),
            "extract_variable": ExtractVariableGenerator(config),
            "replace_magic_numbers": ReplaceMagicNumbersGenerator(config),
            "simplify_conditional": SimplifyConditionalGenerator(config),
            "remove_dead_code": RemoveDeadCodeGenerator(config),
            "extract_class": ExtractClassGenerator(config)
        }
    
    def generate_pattern(self, pattern_type: str, complexity: str = "medium") -> CodePattern:
        """Generate a specific pattern type."""
        if pattern_type not in self.generators:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        return self.generators[pattern_type].generate_pattern(complexity)
    
    def generate_all_patterns(self, samples_per_pattern: int = 10) -> List[CodePattern]:
        """Generate samples for all pattern types."""
        patterns = []
        complexities = ["low", "medium", "high"]
        
        for pattern_type, generator in self.generators.items():
            for complexity in complexities:
                pattern_count = samples_per_pattern // len(complexities)
                for _ in range(pattern_count):
                    pattern = generator.generate_pattern(complexity)
                    patterns.append(pattern)
        
        return patterns
    
    def generate_mixed_patterns(self, total_samples: int) -> List[CodePattern]:
        """Generate mixed patterns with random types and complexities."""
        patterns = []
        pattern_types = list(self.generators.keys())
        complexities = ["low", "medium", "high"]
        complexity_weights = [0.2, 0.6, 0.2]  # Favor medium complexity
        
        for _ in range(total_samples):
            pattern_type = random.choice(pattern_types)
            complexity = random.choices(complexities, weights=complexity_weights)[0]
            pattern = self.generate_pattern(pattern_type, complexity)
            patterns.append(pattern)
        
        return patterns
    
    def save_patterns(self, patterns: List[CodePattern], output_dir: str):
        """Save patterns to files."""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_data = []
        for pattern in patterns:
            json_data.append({
                "original_code": pattern.original_code,
                "refactored_code": pattern.refactored_code,
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "quality_improvement": pattern.quality_improvement,
                "complexity_reduction": pattern.complexity_reduction,
                "difficulty_level": pattern.difficulty_level,
                "tags": pattern.tags or []
            })
        
        with open(os.path.join(output_dir, "code_patterns.json"), "w") as f:
            json.dump(json_data, f, indent=2)
        
        # Save by pattern type
        pattern_groups = {}
        for pattern in patterns:
            if pattern.pattern_type not in pattern_groups:
                pattern_groups[pattern.pattern_type] = []
            pattern_groups[pattern.pattern_type].append(pattern)
        
        for pattern_type, type_patterns in pattern_groups.items():
            type_dir = os.path.join(output_dir, pattern_type)
            os.makedirs(type_dir, exist_ok=True)
            
            for i, pattern in enumerate(type_patterns):
                # Save original code
                with open(os.path.join(type_dir, f"original_{i:03d}.py"), "w") as f:
                    f.write(pattern.original_code)
                
                # Save refactored code
                with open(os.path.join(type_dir, f"refactored_{i:03d}.py"), "w") as f:
                    f.write(pattern.refactored_code)
                
                # Save metadata
                metadata = {
                    "description": pattern.description,
                    "quality_improvement": pattern.quality_improvement,
                    "complexity_reduction": pattern.complexity_reduction,
                    "difficulty_level": pattern.difficulty_level,
                    "tags": pattern.tags or []
                }
                
                with open(os.path.join(type_dir, f"metadata_{i:03d}.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
        
        print(f"Generated {len(patterns)} patterns saved to {output_dir}")
        print(f"Pattern distribution: {dict((pt, len(pg)) for pt, pg in pattern_groups.items())}")
    
    def get_pattern_statistics(self, patterns: List[CodePattern]) -> Dict:
        """Get statistics about generated patterns."""
        from collections import Counter
        
        stats = {
            "total_patterns": len(patterns),
            "pattern_types": Counter(p.pattern_type for p in patterns),
            "difficulty_levels": Counter(p.difficulty_level for p in patterns),
            "average_complexity_reduction": sum(p.complexity_reduction for p in patterns) / len(patterns),
            "quality_metrics": {
                "readability": [],
                "maintainability": [],
                "performance": [],
                "reusability": []
            }
        }
        
        # Collect quality metrics
        for pattern in patterns:
            for metric in stats["quality_metrics"]:
                if metric in pattern.quality_improvement:
                    stats["quality_metrics"][metric].append(pattern.quality_improvement[metric])
        
        # Calculate averages
        for metric in stats["quality_metrics"]:
            values = stats["quality_metrics"][metric]
            if values:
                stats["quality_metrics"][metric] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
            else:
                stats["quality_metrics"][metric] = {"average": 0, "min": 0, "max": 0, "count": 0}
        
        return stats