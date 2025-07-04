"""
Unit tests for the AI models used in code refactoring.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.codebert_model import CodeBERTForRefactoring, CodeBERTTrainer
from models.starcoder_model import StarCoderForRefactoring, StarCoderTrainer
from models.ensemble_model import EnsembleRefactoringModel

class TestCodeBERTModel(unittest.TestCase):
    """Test cases for CodeBERT model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model": {
                "codebert": {
                    "model_name": "microsoft/codebert-base",
                    "max_length": 512,
                    "hidden_size": 768,
                    "dropout": 0.1
                },
                "lora": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1
                }
            }
        }
        
        self.sample_code = """
def calculate_total(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
    return total
"""
        
        self.refactored_code = """
def calculate_total(items):
    return sum(item for item in items if item > 0)
"""
    
    @patch('models.codebert_model.RobertaModel')
    @patch('models.codebert_model.RobertaTokenizer')
    def test_model_initialization(self, mock_tokenizer, mock_model):
        """Test CodeBERT model initialization."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value.config.hidden_size = 768
        
        # Initialize model
        model = CodeBERTForRefactoring(self.config)
        
        # Assertions
        self.assertIsNotNone(model)
        self.assertEqual(model.max_length, 512)
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
    
    @patch('models.codebert_model.RobertaModel')
    @patch('models.codebert_model.RobertaTokenizer')
    def test_forward_pass(self, mock_tokenizer, mock_model):
        """Test forward pass through the model."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 768
        
        # Mock forward pass output
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_model_instance.return_value = mock_output
        
        # Initialize model
        model = CodeBERTForRefactoring(self.config)
        
        # Create sample input
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        # Forward pass
        with patch.object(model.base_model, '__call__', return_value=mock_output):
            output = model.forward(input_ids, attention_mask)
        
        # Assertions
        self.assertIn('generation_logits', output)
        self.assertIn('pattern_logits', output)
        self.assertIn('quality_scores', output)
    
    @patch('models.codebert_model.RobertaModel')
    @patch('models.codebert_model.RobertaTokenizer')
    def test_pattern_prediction(self, mock_tokenizer, mock_model):
        """Test refactoring pattern prediction."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 768
        
        # Mock tokenizer output
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        
        # Mock model output
        mock_output = {
            'pattern_logits': torch.randn(1, 12)
        }
        
        # Initialize model
        model = CodeBERTForRefactoring(self.config)
        
        # Mock forward method
        with patch.object(model, 'forward', return_value=mock_output):
            pattern, confidence = model.predict_refactoring_pattern(self.sample_code)
        
        # Assertions
        self.assertIsInstance(pattern, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    @patch('models.codebert_model.RobertaModel')
    @patch('models.codebert_model.RobertaTokenizer')
    def test_quality_assessment(self, mock_tokenizer, mock_model):
        """Test code quality assessment."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 768
        
        # Mock tokenizer and model outputs
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        
        mock_output = {
            'quality_scores': torch.randn(1, 4),
            'complexity_score': torch.randn(1, 1)
        }
        
        # Initialize model
        model = CodeBERTForRefactoring(self.config)
        
        # Mock forward method
        with patch.object(model, 'forward', return_value=mock_output):
            quality_scores = model.assess_code_quality(self.sample_code)
        
        # Assertions
        self.assertIsInstance(quality_scores, dict)
        self.assertIn('readability', quality_scores)
        self.assertIn('maintainability', quality_scores)
        self.assertIn('performance', quality_scores)
        self.assertIn('complexity', quality_scores)

class TestStarCoderModel(unittest.TestCase):
    """Test cases for StarCoder model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model": {
                "starcoder": {
                    "model_name": "bigcode/starcoder2-3b",
                    "max_length": 1024,
                    "dropout": 0.1
                },
                "lora": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1
                }
            }
        }
        
        self.sample_code = """
def process_data(data):
    result = []
    for item in data:
        if item is not None and item > 0:
            result.append(item * 2)
    return result
"""
    
    @patch('models.starcoder_model.AutoModelForCausalLM')
    @patch('models.starcoder_model.AutoTokenizer')
    def test_model_initialization(self, mock_tokenizer, mock_model):
        """Test StarCoder model initialization."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value.config.hidden_size = 2048
        
        # Initialize model
        model = StarCoderForRefactoring(self.config)
        
        # Assertions
        self.assertIsNotNone(model)
        self.assertEqual(model.max_length, 1024)
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
    
    @patch('models.starcoder_model.AutoModelForCausalLM')
    @patch('models.starcoder_model.AutoTokenizer')
    def test_code_generation(self, mock_tokenizer, mock_model):
        """Test code generation functionality."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 2048
        
        # Mock tokenizer output
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 20)),
            'attention_mask': torch.ones(1, 20)
        }
        mock_tokenizer_instance.decode.return_value = "<REFACTORED_CODE>def process_data(data):\n    return [item * 2 for item in data if item is not None and item > 0]"
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.eos_token_id = 1
        
        # Mock generation
        mock_model_instance.generate.return_value = torch.randint(0, 1000, (1, 30))
        
        # Mock forward output for confidence
        mock_forward_output = {
            'confidence_score': torch.tensor([0.8]),
            'pattern_logits': torch.randn(1, 12)
        }
        
        # Initialize model
        model = StarCoderForRefactoring(self.config)
        
        # Mock forward method
        with patch.object(model, 'forward', return_value=mock_forward_output):
            result = model.generate_refactored_code(self.sample_code)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('refactored_code', result)
        self.assertIn('confidence', result)
        self.assertIn('predicted_pattern', result)
    
    @patch('models.starcoder_model.AutoModelForCausalLM')
    @patch('models.starcoder_model.AutoTokenizer')
    def test_batch_refactoring(self, mock_tokenizer, mock_model):
        """Test batch refactoring functionality."""
        # Setup mocks similar to single refactoring
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 2048
        
        # Initialize model
        model = StarCoderForRefactoring(self.config)
        
        # Mock generate_refactored_code method
        mock_result = {
            'refactored_code': 'def test(): pass',
            'confidence': 0.8,
            'predicted_pattern': 'extract_method'
        }
        
        with patch.object(model, 'generate_refactored_code', return_value=mock_result):
            code_snippets = [self.sample_code, "def another_function(): pass"]
            results = model.batch_refactor(code_snippets)
        
        # Assertions
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('refactored_code', result)
            self.assertIn('confidence', result)

class TestEnsembleModel(unittest.TestCase):
    """Test cases for Ensemble model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model": {
                "ensemble": {
                    "codebert_weight": 0.4,
                    "starcoder_weight": 0.6,
                    "fusion_method": "weighted_average"
                },
                "codebert": {
                    "model_name": "microsoft/codebert-base",
                    "max_length": 512,
                    "dropout": 0.1
                },
                "starcoder": {
                    "model_name": "bigcode/starcoder2-3b",
                    "max_length": 1024,
                    "dropout": 0.1
                },
                "lora": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1
                }
            }
        }
        
        self.sample_code = "def test(): return 42"
    
    @patch('models.ensemble_model.StarCoderForRefactoring')
    @patch('models.ensemble_model.CodeBERTForRefactoring')
    def test_ensemble_initialization(self, mock_codebert, mock_starcoder):
        """Test ensemble model initialization."""
        # Setup mocks
        mock_codebert.return_value = Mock()
        mock_starcoder.return_value = Mock()
        
        # Initialize ensemble
        ensemble = EnsembleRefactoringModel(self.config)
        
        # Assertions
        self.assertIsNotNone(ensemble)
        self.assertEqual(ensemble.codebert_weight, 0.4)
        self.assertEqual(ensemble.starcoder_weight, 0.6)
        self.assertEqual(ensemble.fusion_method, "weighted_average")
    
    @patch('models.ensemble_model.StarCoderForRefactoring')
    @patch('models.ensemble_model.CodeBERTForRefactoring')
    def test_ensemble_code_generation(self, mock_codebert, mock_starcoder):
        """Test ensemble code generation."""
        # Setup mocks
        mock_codebert_instance = Mock()
        mock_starcoder_instance = Mock()
        
        mock_codebert.return_value = mock_codebert_instance
        mock_starcoder.return_value = mock_starcoder_instance
        
        # Mock individual model outputs
        mock_codebert_instance.generate_refactored_code.return_value = "def test(): return 1"
        mock_codebert_instance.assess_code_quality.return_value = {
            'readability': 0.8,
            'maintainability': 0.7
        }
        
        mock_starcoder_instance.generate_refactored_code.return_value = {
            'refactored_code': "def test(): return 2",
            'confidence': 0.9
        }
        mock_starcoder_instance.evaluate_refactoring_quality.return_value = {
            'readability': 0.9,
            'maintainability': 0.8
        }
        
        # Initialize ensemble
        ensemble = EnsembleRefactoringModel(self.config)
        ensemble.codebert_model = mock_codebert_instance
        ensemble.starcoder_model = mock_starcoder_instance
        
        # Mock ensemble-specific methods
        with patch.object(ensemble, '_calculate_ensemble_confidence', return_value=0.85):
            with patch.object(ensemble, '_weighted_selection') as mock_selection:
                mock_selection.return_value = {
                    'refactored_code': 'def test(): return 2',
                    'model': 'starcoder',
                    'quality_scores': {'readability': 0.9}
                }
                
                result = ensemble.generate_refactored_code(self.sample_code)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('refactored_code', result)
        self.assertIn('ensemble_confidence', result)
        self.assertIn('selected_model', result)
    
    @patch('models.ensemble_model.StarCoderForRefactoring')
    @patch('models.ensemble_model.CodeBERTForRefactoring')
    def test_ensemble_evaluation(self, mock_codebert, mock_starcoder):
        """Test ensemble evaluation functionality."""
        # Setup mocks
        mock_codebert_instance = Mock()
        mock_starcoder_instance = Mock()
        
        mock_codebert.return_value = mock_codebert_instance
        mock_starcoder.return_value = mock_starcoder_instance
        
        # Initialize ensemble
        ensemble = EnsembleRefactoringModel(self.config)
        ensemble.codebert_model = mock_codebert_instance
        ensemble.starcoder_model = mock_starcoder_instance
        
        # Mock test data
        test_data = [
            {
                'original_code': 'def old(): pass',
                'refactored_code': 'def new(): pass',
                'pattern_type': 'extract_method'
            }
        ]
        
        # Mock evaluation method
        with patch.object(ensemble, '_codes_equivalent', return_value=True):
            results = ensemble.evaluate_ensemble_performance(test_data)
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertIn('ensemble_accuracy', results)
        self.assertIn('codebert_accuracy', results)
        self.assertIn('starcoder_accuracy', results)


class TestCodeBERTTrainer(unittest.TestCase):
    """Test cases for CodeBERT trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "training": {
                "batch_size": 16,
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "warmup_steps": 100
            },
            "validation": {
                "eval_steps": 50,
                "save_steps": 100,
                "logging_steps": 10
            }
        }
    
    @patch('models.codebert_model.CodeBERTForRefactoring')
    def test_trainer_initialization(self, mock_model):
        """Test trainer initialization."""
        mock_model_instance = Mock()
        trainer = CodeBERTTrainer(mock_model_instance, self.config)
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.config, self.config)
    
    @patch('models.codebert_model.CodeBERTForRefactoring')
    def test_training_args_creation(self, mock_model):
        """Test creation of training arguments."""
        mock_model_instance = Mock()
        trainer = CodeBERTTrainer(mock_model_instance, self.config)
        
        training_args = trainer.create_training_args()
        
        self.assertIsNotNone(training_args)
        self.assertEqual(training_args.per_device_train_batch_size, 16)
        self.assertEqual(training_args.learning_rate, 5e-5)
    
    @patch('models.codebert_model.CodeBERTForRefactoring')
    def test_compute_metrics(self, mock_model):
        """Test metrics computation."""
        mock_model_instance = Mock()
        trainer = CodeBERTTrainer(mock_model_instance, self.config)
        
        # Mock evaluation predictions
        predictions = (
            np.random.randn(10, 100),  # generation_preds
            np.random.randn(10, 12),   # pattern_preds
            np.random.randn(10, 4),    # quality_preds
            np.random.randn(10, 1)     # complexity_preds
        )
        
        labels = (
            np.random.randint(0, 100, (10, 100)),  # generation_labels
            np.random.randint(0, 12, 10),          # pattern_labels
            np.random.randn(10, 4),                # quality_labels
            np.random.randn(10, 1)                 # complexity_labels
        )
        
        eval_pred = Mock()
        eval_pred.predictions = predictions
        eval_pred.label_ids = labels
        
        metrics = trainer.compute_metrics(eval_pred)
        
        self.assertIsInstance(metrics, dict)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "model": {
                "codebert": {
                    "model_name": "microsoft/codebert-base",
                    "max_length": 128,  # Smaller for testing
                    "dropout": 0.1
                },
                "starcoder": {
                    "model_name": "bigcode/starcoder2-3b",
                    "max_length": 128,
                    "dropout": 0.1
                },
                "ensemble": {
                    "codebert_weight": 0.5,
                    "starcoder_weight": 0.5,
                    "fusion_method": "weighted_average"
                },
                "lora": {
                    "r": 4,  # Smaller for testing
                    "lora_alpha": 8,
                    "lora_dropout": 0.1
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skip("Requires actual model files")
    def test_model_save_load_cycle(self):
        """Test saving and loading models."""
        # This test would require actual model weights
        # Skip for unit testing, include in integration tests
        pass
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Test missing required config
        invalid_config = {"model": {}}
        
        with self.assertRaises(KeyError):
            CodeBERTForRefactoring(invalid_config)
    
    @patch('models.codebert_model.RobertaModel')
    @patch('models.codebert_model.RobertaTokenizer')
    def test_error_handling(self, mock_tokenizer, mock_model):
        """Test error handling in model operations."""
        # Setup mocks to raise exceptions
        mock_tokenizer.from_pretrained.side_effect = Exception("Network error")
        
        with self.assertRaises(Exception):
            CodeBERTForRefactoring(self.config)
    
    def test_ensemble_weight_validation(self):
        """Test ensemble weight validation."""
        # Test weights that don't sum to 1
        config = self.config.copy()
        config["model"]["ensemble"]["codebert_weight"] = 0.3
        config["model"]["ensemble"]["starcoder_weight"] = 0.3
        
        # This should still work, weights are normalized internally
        with patch('models.ensemble_model.CodeBERTForRefactoring'), \
             patch('models.ensemble_model.StarCoderForRefactoring'):
            ensemble = EnsembleRefactoringModel(config)
            self.assertIsNotNone(ensemble)


class TestModelUtilities(unittest.TestCase):
    """Test utility functions for models."""
    
    def test_model_size_calculation(self):
        """Test model size calculation utilities."""
        # Mock model for size calculation
        mock_model = Mock()
        mock_model.parameters.return_value = [
            torch.randn(100, 200),  # 20000 parameters
            torch.randn(50)         # 50 parameters
        ]
        
        # This would be implemented in model utilities
        # total_params = sum(p.numel() for p in mock_model.parameters())
        # self.assertEqual(total_params, 20050)
    
    def test_model_device_handling(self):
        """Test model device handling."""
        # Test CUDA availability check
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if cuda_available else "cpu")
        
        self.assertIn(str(device), ["cuda", "cpu"])
    
    @patch('torch.save')
    def test_checkpoint_saving(self, mock_save):
        """Test model checkpoint saving."""
        # Mock saving checkpoint
        checkpoint = {
            'model_state_dict': {'layer.weight': torch.randn(10, 10)},
            'optimizer_state_dict': {},
            'epoch': 5,
            'loss': 0.1
        }
        
        save_path = os.path.join(self.temp_dir, "checkpoint.pt")
        torch.save(checkpoint, save_path)
        
        mock_save.assert_called_once()


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)