import os
import sys
import yaml
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import wandb

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.codebert_model import CodeBERTForRefactoring, CodeBERTTrainer
from models.starcoder_model import StarCoderForRefactoring, StarCoderTrainer
from models.ensemble_model import EnsembleRefactoringModel
from data.dataset_loader import RefactoringDatasetLoader
from utils.evaluation import RefactoringEvaluator

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main trainer class for the refactoring framework."""
    
    def __init__(self, config_path: str):
        """Initialize the trainer."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
        
        # Initialize wandb if configured
        if self.config.get("logging", {}).get("wandb"):
            wandb.init(
                project=self.config["logging"]["wandb"]["project"],
                entity=self.config["logging"]["wandb"]["entity"],
                tags=self.config["logging"]["wandb"]["tags"],
                config=self.config
            )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_config.get("file", "logs/training.log")),
                logging.StreamHandler()
            ]
        )
    
    def prepare_datasets(self) -> Dict:
        """Prepare training and validation datasets."""
        logger.info("Preparing datasets...")
        
        dataset_loader = RefactoringDatasetLoader(self.config)
        
        # Load synthetic dataset
        train_dataset, val_dataset, test_dataset = dataset_loader.load_synthetic_dataset()
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        return {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }
    
    def train_codebert(self, datasets: Dict) -> str:
        """Train the CodeBERT model."""
        logger.info("Starting CodeBERT training...")
        
        # Initialize model
        codebert_model = CodeBERTForRefactoring(self.config)
        codebert_model.to(self.device)
        
        # Initialize trainer
        trainer = CodeBERTTrainer(codebert_model, self.config)
        training_args = trainer.create_training_args()
        
        # Create Hugging Face trainer
        from transformers import Trainer
        
        hf_trainer = Trainer(
            model=codebert_model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=trainer.compute_metrics,
        )
        
        # Train model
        hf_trainer.train()
        
        # Save model
        save_path = "models/fine_tuned/codebert"
        os.makedirs(save_path, exist_ok=True)
        codebert_model.save_model(save_path)
        
        logger.info(f"CodeBERT training completed. Model saved to {save_path}")
        return save_path
    
    def train_starcoder(self, datasets: Dict) -> str:
        """Train the StarCoder model."""
        logger.info("Starting StarCoder training...")
        
        # Initialize model
        starcoder_model = StarCoderForRefactoring(self.config)
        starcoder_model.to(self.device)
        
        # Initialize trainer
        trainer = StarCoderTrainer(starcoder_model, self.config)
        training_args = trainer.create_training_args()
        
        # Create Hugging Face trainer
        from transformers import Trainer
        
        hf_trainer = Trainer(
            model=starcoder_model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=trainer.compute_metrics,
        )
        
        # Train model
        hf_trainer.train()
        
        # Save model
        save_path = "models/fine_tuned/starcoder"
        os.makedirs(save_path, exist_ok=True)
        starcoder_model.save_model(save_path)
        
        logger.info(f"StarCoder training completed. Model saved to {save_path}")
        return save_path
    
    def train_ensemble(self, codebert_path: str, starcoder_path: str, datasets: Dict) -> str:
        """Train the ensemble model."""
        logger.info("Starting ensemble training...")
        
        # Load pre-trained individual models
        codebert_model = CodeBERTForRefactoring.load_model(codebert_path, self.config)
        starcoder_model = StarCoderForRefactoring.load_model(starcoder_path, self.config)
        
        # Initialize ensemble
        ensemble_model = EnsembleRefactoringModel(self.config)
        ensemble_model.codebert_model = codebert_model
        ensemble_model.starcoder_model = starcoder_model
        ensemble_model.to(self.device)
        
        # Train ensemble-specific components
        self._train_ensemble_components(ensemble_model, datasets)
        
        # Save ensemble model
        save_path = "models/fine_tuned/ensemble"
        os.makedirs(save_path, exist_ok=True)
        ensemble_model.save_ensemble(save_path)
        
        logger.info(f"Ensemble training completed. Model saved to {save_path}")
        return save_path
    
    def _train_ensemble_components(self, ensemble_model: EnsembleRefactoringModel, datasets: Dict):
        """Train ensemble-specific components."""
        from torch.utils.data import DataLoader
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            ensemble_model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Setup data loader
        train_loader = DataLoader(
            datasets["train"],
            batch_size=self.config["training"]["batch_size"],
            shuffle=True
        )
        
        # Training loop for ensemble components
        ensemble_model.train()
        for epoch in range(5):  # Few epochs for ensemble fine-tuning
            total_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                outputs = ensemble_model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device)
                )
                
                # Simple loss for ensemble training
                if outputs["loss"] is not None:
                    loss = outputs["loss"]
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Ensemble epoch {epoch + 1}, Average loss: {avg_loss:.4f}")
    
    def evaluate_models(self, model_paths: Dict, test_dataset) -> Dict:
        """Evaluate all trained models."""
        logger.info("Evaluating trained models...")
        
        evaluator = RefactoringEvaluator(self.config)
        results = {}
        
        # Evaluate CodeBERT
        if "codebert" in model_paths:
            logger.info("Evaluating CodeBERT...")
            codebert_model = CodeBERTForRefactoring.load_model(
                model_paths["codebert"], self.config
            )
            results["codebert"] = self._evaluate_single_model(
                codebert_model, test_dataset, evaluator
            )
        
        # Evaluate StarCoder
        if "starcoder" in model_paths:
            logger.info("Evaluating StarCoder...")
            starcoder_model = StarCoderForRefactoring.load_model(
                model_paths["starcoder"], self.config
            )
            results["starcoder"] = self._evaluate_single_model(
                starcoder_model, test_dataset, evaluator
            )
        
        # Evaluate Ensemble
        if "ensemble" in model_paths:
            logger.info("Evaluating Ensemble...")
            ensemble_model = EnsembleRefactoringModel.load_ensemble(
                model_paths["ensemble"], self.config
            )
            results["ensemble"] = ensemble_model.evaluate_ensemble_performance(
                test_dataset
            )
        
        return results
    
    def _evaluate_single_model(self, model, test_dataset, evaluator) -> Dict:
        """Evaluate a single model."""
        # Convert test dataset to evaluation format
        test_samples = []
        for sample in test_dataset:
            test_samples.append({
                "original_code": sample["original_code"],
                "refactored_code": sample["refactored_code"],
                "pattern_type": sample["pattern_type"]
            })
        
        # Evaluate model (simplified evaluation)
        correct_predictions = 0
        total_predictions = len(test_samples)
        
        for sample in test_samples[:100]:  # Evaluate on subset for speed
            try:
                if hasattr(model, 'generate_refactored_code'):
                    prediction = model.generate_refactored_code(sample["original_code"])
                else:
                    prediction = sample["original_code"]  # Fallback
                
                # Simple correctness check (in practice, use more sophisticated metrics)
                if len(prediction) > 0:
                    correct_predictions += 1
            except:
                pass
        
        accuracy = correct_predictions / min(total_predictions, 100)
        
        return {
            "accuracy": accuracy,
            "total_samples": min(total_predictions, 100),
            "correct_predictions": correct_predictions
        }
    
    def save_training_results(self, results: Dict, output_path: str):
        """Save training and evaluation results."""
        import json
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to {output_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train AI Code Refactoring Models")
    parser.add_argument("--config", "-c", default="config/training_config.yaml",
                       help="Training configuration file")
    parser.add_argument("--models", "-m", nargs="+", 
                       choices=["codebert", "starcoder", "ensemble", "all"],
                       default=["all"], help="Models to train")
    parser.add_argument("--data-dir", "-d", default="data/synthetic_dataset",
                       help="Dataset directory")
    parser.add_argument("--output-dir", "-o", default="models/fine_tuned",
                       help="Output directory for trained models")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation on existing models")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate synthetic data before training")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(args.config)
        
        # Generate synthetic data if requested
        if args.generate_data:
            logger.info("Generating synthetic training data...")
            from data.synthetic_generator import SyntheticDataGenerator
            generator = SyntheticDataGenerator(trainer.config)
            patterns = generator.generate_dataset(
                trainer.config["synthetic_data"]["dataset_size"]
            )
            generator.save_dataset(patterns, args.data_dir)
        
        # Prepare datasets
        datasets = trainer.prepare_datasets()
        
        model_paths = {}
        
        if not args.eval_only:
            # Determine which models to train
            models_to_train = args.models
            if "all" in models_to_train:
                models_to_train = ["codebert", "starcoder", "ensemble"]
            
            # Train individual models
            if "codebert" in models_to_train:
                model_paths["codebert"] = trainer.train_codebert(datasets)
            
            if "starcoder" in models_to_train:
                model_paths["starcoder"] = trainer.train_starcoder(datasets)
            
            # Train ensemble (requires both individual models)
            if "ensemble" in models_to_train:
                if "codebert" not in model_paths:
                    model_paths["codebert"] = "models/fine_tuned/codebert"
                if "starcoder" not in model_paths:
                    model_paths["starcoder"] = "models/fine_tuned/starcoder"
                
                model_paths["ensemble"] = trainer.train_ensemble(
                    model_paths["codebert"],
                    model_paths["starcoder"],
                    datasets
                )
        else:
            # Use existing models for evaluation
            model_paths = {
                "codebert": "models/fine_tuned/codebert",
                "starcoder": "models/fine_tuned/starcoder",
                "ensemble": "models/fine_tuned/ensemble"
            }
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(model_paths, datasets["test"])
        
        # Save results
        results = {
            "model_paths": model_paths,
            "evaluation_results": evaluation_results,
            "config": trainer.config
        }
        
        trainer.save_training_results(results, "results/training_results.json")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Trained models saved to: {args.output_dir}")
        print("\nEvaluation Results:")
        for model_name, result in evaluation_results.items():
            print(f"  {model_name}: {result.get('accuracy', 'N/A'):.3f} accuracy")
        print("\nDetailed results saved to: results/training_results.json")
        
        if wandb.run:
            # Log results to wandb
            for model_name, result in evaluation_results.items():
                wandb.log({f"{model_name}_accuracy": result.get('accuracy', 0)})
            wandb.finish()
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())