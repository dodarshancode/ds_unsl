#!/usr/bin/env python3
"""
Test script to validate the fine-tuning setup before running the main training
This script checks all dependencies, GPU availability, and data format
"""

import os
import sys
import json
import pandas as pd
import torch
import subprocess
from pathlib import Path

def check_gpu_setup():
    """Check GPU availability and configuration"""
    print("🔍 Checking GPU Setup...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Please install PyTorch with CUDA support.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPU(s)")
    
    if gpu_count < 4:
        print(f"⚠️  Warning: Expected 4 GPUs, found {gpu_count}")
    
    # Check GPU memory
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        memory_gb = gpu_props.total_memory / (1024**3)
        print(f"   GPU {i}: {gpu_props.name} ({memory_gb:.1f} GB)")
        
        if memory_gb < 10:
            print(f"⚠️  Warning: GPU {i} has less than 10GB memory")
    
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        "torch", "transformers", "datasets", "accelerate", 
        "peft", "deepspeed", "pandas", "numpy", "sklearn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    # Check Unsloth separately (special installation)
    try:
        import unsloth
        print("✅ unsloth")
    except ImportError:
        print("❌ unsloth - NOT FOUND")
        missing_packages.append("unsloth")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_deepspeed():
    """Check DeepSpeed installation and configuration"""
    print("\n⚡ Checking DeepSpeed...")
    
    try:
        import deepspeed
        print(f"✅ DeepSpeed version: {deepspeed.__version__}")
        
        # Test DeepSpeed report
        try:
            result = subprocess.run(['ds_report'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ DeepSpeed report successful")
            else:
                print("⚠️  DeepSpeed report failed, but package is installed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  Could not run ds_report command")
        
        return True
    except ImportError:
        print("❌ DeepSpeed not found")
        return False

def check_data_format(csv_path):
    """Check if the CSV data is in the correct format"""
    print(f"\n📊 Checking Data Format: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded successfully ({len(df)} rows)")
        
        required_columns = ['Instruction', 'Output']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            print(f"   Found columns: {list(df.columns)}")
            return False
        
        print("✅ Required columns found: Instruction, Output")
        
        # Check for empty values
        empty_instructions = df['Instruction'].isna().sum() + (df['Instruction'] == '').sum()
        empty_outputs = df['Output'].isna().sum() + (df['Output'] == '').sum()
        
        if empty_instructions > 0:
            print(f"⚠️  Found {empty_instructions} empty instructions")
        if empty_outputs > 0:
            print(f"⚠️  Found {empty_outputs} empty outputs")
        
        # Show sample data
        print("\n📝 Sample data:")
        for i in range(min(2, len(df))):
            instruction = df.iloc[i]['Instruction'][:100] + "..." if len(df.iloc[i]['Instruction']) > 100 else df.iloc[i]['Instruction']
            output = df.iloc[i]['Output'][:100] + "..." if len(df.iloc[i]['Output']) > 100 else df.iloc[i]['Output']
            print(f"   Row {i+1}:")
            print(f"     Instruction: {instruction}")
            print(f"     Output: {output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV: {str(e)}")
        return False

def check_config_files():
    """Check if configuration files exist"""
    print("\n⚙️  Checking Configuration Files...")
    
    config_files = {
        'ds_config.json': 'DeepSpeed configuration',
        'requirements.txt': 'Python requirements',
        'train.py': 'Training script'
    }
    
    all_present = True
    for file_name, description in config_files.items():
        if os.path.exists(file_name):
            print(f"✅ {file_name} ({description})")
        else:
            print(f"❌ {file_name} ({description}) - NOT FOUND")
            all_present = False
    
    return all_present

def create_sample_data():
    """Create a sample CSV file for testing"""
    print("\n📝 Creating sample data...")
    
    sample_data = {
        'Instruction': [
            'Write a function to calculate the factorial of a number',
            'Create a class to represent a binary tree node',
            'Write a function to reverse a string',
            'Implement a simple bubble sort algorithm',
            'Create a function to check if a number is prime'
        ],
        'Output': [
            'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)',
            'class TreeNode:\n    def __init__(self, val=0):\n        self.val = val\n        self.left = None\n        self.right = None',
            'def reverse_string(s):\n    return s[::-1]',
            'def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr',
            'def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    print("✅ Created sample_data.csv for testing")
    return True

def test_model_loading():
    """Test if we can load the base model"""
    print("\n🤖 Testing Model Loading...")
    
    try:
        from unsloth import FastLanguageModel
        
        print("   Loading CodeLlama model (this may take a few minutes)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="codellama/CodeLlama-7b-Instruct-hf",
            max_seq_length=512,  # Small for testing
            dtype=None,
            load_in_4bit=True,
        )
        
        print("✅ Model loaded successfully")
        
        # Test tokenization
        test_text = "def hello_world():"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test passed (tokens: {len(tokens['input_ids'][0])})")
        
        # Clean up memory
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

def run_memory_test():
    """Test GPU memory usage"""
    print("\n💾 Testing GPU Memory...")
    
    try:
        # Allocate some memory to test
        device = torch.device("cuda:0")
        
        # Test memory allocation
        test_tensor = torch.randn(1000, 1000, device=device)
        memory_used = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"✅ GPU memory test passed ({memory_used:.2f} GB allocated)")
        
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU memory test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 CodeLlama Fine-tuning Setup Validation\n")
    print("=" * 50)
    
    tests = [
        ("GPU Setup", check_gpu_setup),
        ("Dependencies", check_dependencies),
        ("DeepSpeed", check_deepspeed),
        ("Configuration Files", check_config_files),
        ("GPU Memory", run_memory_test),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {str(e)}")
    
    # Data format test (optional)
    csv_path = input(f"\n📊 Enter path to your CSV file (or press Enter to skip): ").strip()
    if csv_path:
        if check_data_format(csv_path):
            passed_tests += 1
        total_tests += 1
    else:
        print("⏭️  Skipping data format check")
        create_sample_data()
    
    # Model loading test (optional but recommended)
    test_model = input(f"\n🤖 Test model loading? This will download ~13GB (y/N): ").strip().lower()
    if test_model in ['y', 'yes']:
        if test_model_loading():
            passed_tests += 1
        total_tests += 1
    else:
        print("⏭️  Skipping model loading test")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your setup is ready for fine-tuning.")
        print("\nNext steps:")
        print("1. Prepare your training data in CSV format")
        print("2. Run: deepspeed --num_gpus=4 train.py --csv_path your_data.csv")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        print("Check the error messages above for specific problems.")
    
    print("\n🔗 For help, refer to the setup guide or create an issue.")

if __name__ == "__main__":
    main()
