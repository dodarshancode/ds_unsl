#!/usr/bin/env python3
"""
Memory-optimized CodeLlama fine-tuning with Unsloth
Designed for small datasets and limited GPU memory
"""

import os
import sys
import json
import logging
import argparse
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_seq_length: int = 1024  # Reduced from 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True
    
@dataclass
class LoRAConfig:
    """Configuration for LoRA parameters"""
    r: int = 16  # Reduced from 64 for memory efficiency
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_alpha: int = 32  # Increased to compensate for lower rank
    lora_dropout: float = 0.1  # Added dropout for better generalization
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict] = None

@dataclass
class DataConfig:
    """Configuration for data processing"""
    csv_path: str = "training_data.csv"
    instruction_column: str = "Instruction"
    output_column: str = "Output"
    test_size: float = 0.1
    validation_size: float = 0.1
    max_length: int = 1024  # Reduced from 2048
    random_state: int = 42

def detect_device_capabilities():
    """Detect device capabilities for optimal configuration"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (will be very slow)")
        return {"device": "cpu", "fp16": False, "bf16": False}
    
    device_name = torch.cuda.get_device_name()
    logger.info(f"GPU detected: {device_name}")
    
    # Check FP16 support
    fp16_supported = True
    try:
        # Test FP16 capability
        test_tensor = torch.randn(10, 10, dtype=torch.float16, device='cuda')
        _ = torch.matmul(test_tensor, test_tensor)
    except:
        fp16_supported = False
    
    # Check BF16 support
    bf16_supported = torch.cuda.is_bf16_supported()
    
    logger.info(f"Device capabilities - FP16: {fp16_supported}, BF16: {bf16_supported}")
    
    return {
        "device": "cuda",
        "fp16": fp16_supported,
        "bf16": bf16_supported,
        "device_name": device_name
    }

class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def load_and_split_data(self) -> tuple:
        """Load CSV data and split into train/val/test sets"""
        logger.info(f"Loading data from {self.config.csv_path}")
        
        if not os.path.exists(self.config.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.config.csv_path}")
            
        df = pd.read_csv(self.config.csv_path)
        
        # Validate columns
        required_cols = [self.config.instruction_column, self.config.output_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
            
        # Clean data
        df = df.dropna(subset=required_cols)
        df = df[df[self.config.instruction_column].str.strip() != ""]
        df = df[df[self.config.output_column].str.strip() != ""]
        
        logger.info(f"Loaded {len(df)} samples after cleaning")
        
        # For small datasets, ensure minimum samples in each split
        if len(df) < 50:
            logger.warning("Very small dataset detected. Using minimal validation split.")
            train_df = df.iloc[:-5] if len(df) > 10 else df
            val_df = df.iloc[-5:] if len(df) > 10 else df.sample(min(3, len(df)))
            test_df = pd.DataFrame()
        else:
            # Split data
            train_df, temp_df = train_test_split(
                df, test_size=self.config.test_size + self.config.validation_size,
                random_state=self.config.random_state
            )
            
            if self.config.validation_size > 0 and len(temp_df) > 5:
                val_size = self.config.validation_size / (self.config.test_size + self.config.validation_size)
                val_df, test_df = train_test_split(
                    temp_df, test_size=1-val_size,
                    random_state=self.config.random_state
                )
            else:
                val_df = temp_df
                test_df = pd.DataFrame()
            
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def format_prompts(self, examples: Dict) -> Dict:
        """Format examples into instruction-following format"""
        instructions = examples[self.config.instruction_column]
        outputs = examples[self.config.output_column]
        
        texts = []
        for instruction, output in zip(instructions, outputs):
            # Simplified prompt format for better memory usage
            text = f"""### Instruction:
{instruction}

### Response:
{output}"""
            texts.append(text)
        
        return {"text": texts}
    
    def create_datasets(self) -> DatasetDict:
        """Create Hugging Face datasets"""
        train_df, val_df, test_df = self.load_and_split_data()
        
        datasets = {}
        
        # Create train dataset
        train_dataset = Dataset.from_pandas(train_df)
        train_dataset = train_dataset.map(
            self.format_prompts,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        datasets["train"] = train_dataset
        
        # Create validation dataset
        if len(val_df) > 0:
            val_dataset = Dataset.from_pandas(val_df)
            val_dataset = val_dataset.map(
                self.format_prompts,
                batched=True,
                remove_columns=val_dataset.column_names
            )
            datasets["validation"] = val_dataset
        
        # Create test dataset
        if len(test_df) > 0:
            test_dataset = Dataset.from_pandas(test_df)
            test_dataset = test_dataset.map(
                self.format_prompts,
                batched=True,
                remove_columns=test_dataset.column_names
            )
            datasets["test"] = test_dataset
            
        return DatasetDict(datasets)

class MetricsCallback(TrainerCallback):
    """Custom callback for logging metrics"""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            if "train_loss" in logs:
                logger.info(f"Step {state.global_step}: Train Loss = {logs['train_loss']:.4f}")
            if "eval_loss" in logs:
                logger.info(f"Step {state.global_step}: Eval Loss = {logs['eval_loss']:.4f}")

class CodeLlamaTrainer:
    """Main trainer class"""
    
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig, data_config: DataConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.data_config = data_config
        self.model = None
        self.tokenizer = None
        self.device_caps = detect_device_capabilities()
        
    def setup_model(self):
        """Initialize model and tokenizer with Unsloth optimizations"""
        logger.info("Setting up model and tokenizer...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.model_name,
            max_seq_length=self.model_config.max_seq_length,
            dtype=self.model_config.dtype,
            load_in_4bit=self.model_config.load_in_4bit,
        )
        
        # Setup LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            random_state=self.lora_config.random_state,
            use_rslora=self.lora_config.use_rslora,
            loftq_config=self.lora_config.loftq_config,
        )
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model setup completed")
        
    def create_training_args(self, output_dir: str, num_train_samples: int) -> TrainingArguments:
        """Create memory-optimized training arguments"""
        
        # Calculate optimal steps based on dataset size
        steps_per_epoch = max(1, num_train_samples // 1)  # Very small batch size
        total_epochs = min(10, max(3, 1000 // num_train_samples))  # More epochs for small datasets
        max_steps = steps_per_epoch * total_epochs
        
        logger.info(f"Training configuration: {total_epochs} epochs, {max_steps} steps")
        
        return TrainingArguments(
            per_device_train_batch_size=1,  # Minimal batch size
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Simulate larger batch size
            warmup_steps=min(50, max_steps // 10),
            max_steps=max_steps,
            learning_rate=5e-4,  # Slightly higher LR for small datasets
            fp16=self.device_caps["fp16"] and not self.device_caps["bf16"],
            bf16=self.device_caps["bf16"],
            logging_steps=max(1, max_steps // 20),
            eval_steps=max(10, max_steps // 5),
            save_steps=max(25, max_steps // 4),
            evaluation_strategy="steps" if num_train_samples > 20 else "no",
            save_strategy="steps",
            output_dir=output_dir,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # Better for small datasets
            seed=3407,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            load_best_model_at_end=True if num_train_samples > 20 else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,  # Save disk space
            report_to=[],  # Disable wandb/tensorboard for simplicity
            run_name="codellama-finetune",
            dataloader_pin_memory=False,  # Reduce memory usage
            ddp_find_unused_parameters=False,
        )
    
    def train(self, output_dir: str = "./results"):
        """Main training loop"""
        logger.info("Starting training process...")
        
        # Setup model
        self.setup_model()
        
        # Process data
        data_processor = DataProcessor(self.data_config)
        datasets = data_processor.create_datasets()
        
        num_train_samples = len(datasets["train"])
        logger.info(f"Training on {num_train_samples} samples")
        
        # Create training arguments
        training_args = self.create_training_args(output_dir, num_train_samples)
        
        # Data collator with padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8  # Efficiency optimization
        )
        
        # Initialize trainer
        callbacks = [MetricsCallback()]
        if "validation" in datasets and len(datasets["validation"]) > 5:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))
        
        trainer = UnslothTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation") if len(datasets.get("validation", [])) > 5 else None,
            dataset_text_field="text",
            max_seq_length=self.model_config.max_seq_length,
            data_collator=data_collator,
            args=training_args,
            callbacks=callbacks,
        )
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Start training
        logger.info("Beginning training...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("Out of memory error. Try reducing max_seq_length or use a smaller model.")
                logger.error("Consider using CodeLlama-1b instead of 7b for your hardware.")
            raise e
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        if trainer.state.log_history:
            with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
                
        logger.info("Training completed successfully!")
        
        # Test the model with a sample
        self.test_model_sample(output_dir)
    
    def test_model_sample(self, model_path: str):
        """Test the trained model with a sample"""
        logger.info("Testing trained model...")
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        
        # Test prompt
        test_instruction = "Write a Python function to calculate fibonacci numbers"
        test_prompt = f"""### Instruction:
{test_instruction}

### Response:
"""
        
        inputs = self.tokenizer([test_prompt], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(test_prompt):].strip()
        
        logger.info(f"Sample generation:\nInput: {test_instruction}\nOutput: {generated_text[:200]}...")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune CodeLlama with Unsloth (Memory Optimized)")
    
    # Model arguments
    parser.add_argument("--model_name", default="codellama/CodeLlama-7b-Instruct-hf", 
                       help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=1024, 
                       help="Maximum sequence length")
    
    # Data arguments
    parser.add_argument("--csv_path", required=True, 
                       help="Path to CSV training data")
    parser.add_argument("--instruction_column", default="Instruction", 
                       help="Name of instruction column")
    parser.add_argument("--output_column", default="Output", 
                       help="Name of output column")
    parser.add_argument("--test_size", type=float, default=0.1, 
                       help="Test set size")
    parser.add_argument("--validation_size", type=float, default=0.1, 
                       help="Validation set size")
    
    # Training arguments
    parser.add_argument("--output_dir", default="./results", 
                       help="Output directory for model and logs")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, 
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, 
                       help="LoRA dropout")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory < 8:
            logger.warning("Low GPU memory detected. Consider using a smaller model or reducing sequence length.")
    
    # Initialize configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length
    )
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    data_config = DataConfig(
        csv_path=args.csv_path,
        instruction_column=args.instruction_column,
        output_column=args.output_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        max_length=args.max_seq_length
    )
    
    # Initialize trainer
    trainer = CodeLlamaTrainer(model_config, lora_config, data_config)
    
    # Start training
    try:
        trainer.train(args.output_dir)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if "out of memory" in str(e).lower():
            logger.error("\nSuggested fixes for OOM:")
            logger.error("1. Use --max_seq_length 512")
            logger.error("2. Use CodeLlama-1b-Instruct instead of 7b")
            logger.error("3. Reduce LoRA rank: --lora_r 8")
        sys.exit(1)

if __name__ == "__main__":
    main()
