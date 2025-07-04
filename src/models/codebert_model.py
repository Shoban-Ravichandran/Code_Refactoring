# src/models/codebert_model.py
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, RobertaTokenizer, RobertaModel
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CodeBERTForRefactoring(nn.Module):
    """CodeBERT model fine-tuned for code refactoring tasks."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.model_name = config["model"]["codebert"]["model_name"]
        self.max_length = config["model"]["codebert"]["max_length"]
        
        # Load base model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.base_model = RobertaModel.from_pretrained(self.model_name)
        
        # Add special tokens for code refactoring
        special_tokens = {
            "additional_special_tokens": [
                "<ORIGINAL>", "<REFACTORED>", "<PATTERN>", "<QUALITY>",
                "<METHOD>", "<CLASS>", "<VARIABLE>", "<FUNCTION>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA for efficient fine-tuning
        self.setup_lora()
        
        # Classification heads for different refactoring tasks
        hidden_size = self.base_model.config.hidden_size
        self.pattern_classifier = nn.Linear(hidden_size, 12)  # 12 refactoring patterns
        self.quality_regressor = nn.Linear(hidden_size, 4)   # 4 quality metrics
        self.complexity_regressor = nn.Linear(hidden_size, 1) # complexity score
        
        # Code generation head
        self.code_generator = nn.Linear(hidden_size, len(self.tokenizer))
        
        self.dropout = nn.Dropout(config["model"]["codebert"]["dropout"])
        
    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config["model"]["lora"]["r"],
            lora_alpha=self.config["model"]["lora"]["lora_alpha"],
            lora_dropout=self.config["model"]["lora"]["lora_dropout"],
            target_modules=["query", "key", "value", "dense"]
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        logger.info("LoRA configuration applied to CodeBERT model")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pattern_labels: Optional[torch.Tensor] = None,
        quality_labels: Optional[torch.Tensor] = None,
        complexity_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get pooled representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Classification and regression heads
        pattern_logits = self.pattern_classifier(pooled_output)
        quality_scores = self.quality_regressor(pooled_output)
        complexity_score = self.complexity_regressor(pooled_output)
        
        # Code generation
        sequence_output = self.dropout(outputs.last_hidden_state)
        generation_logits = self.code_generator(sequence_output)
        
        loss = None
        if labels is not None:
            # Calculate combined loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Generation loss
            shift_logits = generation_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            generation_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Pattern classification loss
            pattern_loss = 0
            if pattern_labels is not None:
                pattern_loss_fct = nn.CrossEntropyLoss()
                pattern_loss = pattern_loss_fct(pattern_logits, pattern_labels)
            
            # Quality regression loss
            quality_loss = 0
            if quality_labels is not None:
                quality_loss_fct = nn.MSELoss()
                quality_loss = quality_loss_fct(quality_scores, quality_labels)
            
            # Complexity regression loss
            complexity_loss = 0
            if complexity_labels is not None:
                complexity_loss_fct = nn.MSELoss()
                complexity_loss = complexity_loss_fct(
                    complexity_score.squeeze(), complexity_labels
                )
            
            # Combine losses with weights
            loss = (0.5 * generation_loss + 
                   0.2 * pattern_loss + 
                   0.2 * quality_loss + 
                   0.1 * complexity_loss)
        
        return {
            "loss": loss,
            "generation_logits": generation_logits,
            "pattern_logits": pattern_logits,
            "quality_scores": quality_scores,
            "complexity_score": complexity_score,
            "hidden_states": outputs.last_hidden_state,
            "pooled_output": pooled_output
        }
    
    def generate_refactored_code(
        self,
        original_code: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 0.8
    ) -> str:
        """Generate refactored code given original code."""
        
        # Tokenize input
        inputs = self.tokenizer(
            f"<ORIGINAL> {original_code} <REFACTORED>",
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            # Generate tokens
            generated_ids = self.base_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            
            # Extract refactored code
            if "<REFACTORED>" in generated_text:
                refactored_code = generated_text.split("<REFACTORED>")[-1].strip()
            else:
                refactored_code = generated_text.strip()
        
        return refactored_code
    
    def predict_refactoring_pattern(self, code: str) -> Tuple[str, float]:
        """Predict the best refactoring pattern for given code."""
        
        inputs = self.tokenizer(
            code,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            pattern_probs = torch.softmax(outputs["pattern_logits"], dim=-1)
            pattern_idx = torch.argmax(pattern_probs, dim=-1).item()
            confidence = pattern_probs[0, pattern_idx].item()
            
            pattern_names = [
                "extract_method", "inline_method", "extract_variable", "inline_variable",
                "move_method", "rename_method", "extract_class", "inline_class",
                "replace_conditional_with_polymorphism", "replace_magic_numbers",
                "remove_dead_code", "simplify_conditional"
            ]
            
            return pattern_names[pattern_idx], confidence
    
    def assess_code_quality(self, code: str) -> Dict[str, float]:
        """Assess code quality metrics."""
        
        inputs = self.tokenizer(
            code,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            quality_scores = outputs["quality_scores"].squeeze().cpu().numpy()
            complexity_score = outputs["complexity_score"].squeeze().cpu().item()
            
            return {
                "readability": float(quality_scores[0]),
                "maintainability": float(quality_scores[1]),
                "performance": float(quality_scores[2]),
                "reusability": float(quality_scores[3]),
                "complexity": complexity_score
            }
    
    def save_model(self, save_directory: str):
        """Save the fine-tuned model."""
        self.base_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # Save additional components
        torch.save({
            "pattern_classifier": self.pattern_classifier.state_dict(),
            "quality_regressor": self.quality_regressor.state_dict(),
            "complexity_regressor": self.complexity_regressor.state_dict(),
            "code_generator": self.code_generator.state_dict(),
            "config": self.config
        }, f"{save_directory}/additional_components.pt")
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def load_model(cls, load_directory: str, config: Dict):
        """Load a fine-tuned model."""
        model = cls(config)
        
        # Load base model and tokenizer
        model.base_model = RobertaModel.from_pretrained(load_directory)
        model.tokenizer = RobertaTokenizer.from_pretrained(load_directory)
        
        # Load additional components
        checkpoint = torch.load(f"{load_directory}/additional_components.pt")
        model.pattern_classifier.load_state_dict(checkpoint["pattern_classifier"])
        model.quality_regressor.load_state_dict(checkpoint["quality_regressor"])
        model.complexity_regressor.load_state_dict(checkpoint["complexity_regressor"])
        model.code_generator.load_state_dict(checkpoint["code_generator"])
        
        logger.info(f"Model loaded from {load_directory}")
        return model

class CodeBERTTrainer:
    """Trainer class for CodeBERT model."""
    
    def __init__(self, model: CodeBERTForRefactoring, config: Dict):
        self.model = model
        self.config = config
        
    def create_training_args(self) -> TrainingArguments:
        """Create training arguments."""
        return TrainingArguments(
            output_dir="./models/codebert_checkpoints",
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["batch_size"],
            learning_rate=self.config["training"]["learning_rate"],
            num_train_epochs=self.config["training"]["num_epochs"],
            warmup_steps=self.config["training"]["warmup_steps"],
            weight_decay=self.config["training"]["weight_decay"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            fp16=self.config["training"]["mixed_precision"],
            gradient_checkpointing=self.config["training"]["gradient_checkpointing"],
            evaluation_strategy="steps",
            eval_steps=self.config["validation"]["eval_steps"],
            save_steps=self.config["validation"]["save_steps"],
            logging_steps=self.config["validation"]["logging_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            report_to="wandb",
            run_name="codebert-refactoring"
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Extract different predictions
        generation_preds = predictions[0]
        pattern_preds = predictions[1]
        quality_preds = predictions[2]
        complexity_preds = predictions[3]
        
        # Extract corresponding labels
        generation_labels = labels[0]
        pattern_labels = labels[1]
        quality_labels = labels[2]
        complexity_labels = labels[3]
        
        metrics = {}
        
        # Pattern classification accuracy
        if pattern_labels is not None:
            pattern_accuracy = np.mean(
                np.argmax(pattern_preds, axis=-1) == pattern_labels
            )
            metrics["pattern_accuracy"] = pattern_accuracy
        
        # Quality prediction MSE
        if quality_labels is not None:
            quality_mse = np.mean((quality_preds - quality_labels) ** 2)
            metrics["quality_mse"] = quality_mse
        
        # Complexity prediction MSE
        if complexity_labels is not None:
            complexity_mse = np.mean((complexity_preds - complexity_labels) ** 2)
            metrics["complexity_mse"] = complexity_mse
        
        return metrics