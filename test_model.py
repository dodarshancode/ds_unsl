#!/usr/bin/env python3
"""
Test script for the fine-tuned CodeLlama model
Use this to test your model after training is complete
"""

import argparse
import torch
from unsloth import FastLanguageModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeLlamaInference:
    """Inference wrapper for fine-tuned CodeLlama model"""
    
    def __init__(self, model_path: str, max_seq_length: int = 2048):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned model"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable fast inference
            FastLanguageModel.for_inference(self.model)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def generate_code(self, instruction: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        """Generate code based on instruction"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Format prompt in the same way as training
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful coding assistant. Generate code based on the given instruction.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize input
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract only the assistant's response
        assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
        if assistant_start in full_output:
            response = full_output.split(assistant_start)[-1].strip()
            # Remove any trailing special tokens
            response = response.replace("<|eot_id|>", "").strip()
            return response
        else:
            return full_output
    
    def interactive_mode(self):
        """Run interactive mode for testing"""
        print("🤖 CodeLlama Interactive Mode")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 50)
        
        while True:
            try:
                instruction = input("\n📝 Enter your coding instruction: ").strip()
                
                if instruction.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not instruction:
                    continue
                
                print("\n🔄 Generating code...")
                response = self.generate_code(instruction)
                
                print("\n💻 Generated Code:")
                print("-" * 30)
                print(response)
                print("-" * 30)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

def run_test_cases(inference_engine: CodeLlamaInference):
    """Run predefined test cases"""
    test_cases = [
        "Write a function to calculate factorial of a number",
        "Create a class for a binary search tree",
        "Write a function to reverse a string",
        "Implement bubble sort algorithm",
        "Create a function to check if a number is prime",
        "Write a function to find the maximum element in a list",
        "Create a class for a simple calculator",
        "Write a function to count vowels in a string"
    ]
    
    print("🧪 Running Test Cases")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case}")
        print("-" * 40)
        
        try:
            response = inference_engine.generate_code(test_case)
            print(response)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)
        
        # Ask user if they want to continue
        if i < len(test_cases):
            continue_test = input("\nContinue to next test? (Y/n): ").strip().lower()
            if continue_test == 'n':
                break

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test fine-tuned CodeLlama model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--test_cases", action="store_true", help="Run predefined test cases")
    parser.add_argument("--instruction", type=str, help="Single instruction to test")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = CodeLlamaInference(args.model_path, args.max_seq_length)
    
    try:
        # Load model
        inference_engine.load_model()
        
        if args.interactive:
            # Interactive mode
            inference_engine.interactive_mode()
            
        elif args.test_cases:
            # Run test cases
            run_test_cases(inference_engine)
            
        elif args.instruction:
            # Single instruction
            print(f"📝 Instruction: {args.instruction}")
            print("💻 Generated Code:")
            print("-" * 40)
            response = inference_engine.generate_code(
                args.instruction, 
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            print(response)
            print("-" * 40)
            
        else:
            print("❌ Please specify --interactive, --test_cases, or --instruction")
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
