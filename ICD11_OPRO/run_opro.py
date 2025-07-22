#!/usr/bin/env python3
"""
ICD-11 OPRO Main Execution Script

This script provides a user-friendly interface to run OPRO optimization or evaluate prompts.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import argparse

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from opro.optimize_icd11_prompt import OPROOptimizer, OptimizationResult
from evaluate_prompt import PromptEvaluator, EvaluationResult, evaluate_prompt_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OPROInterface:
    """Main interface class, handles user interaction and system operations"""
    
    def __init__(self):
        """Initialize interface"""
        self.config = self._load_config()
        self.optimizer = None
        self.evaluator = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("❌ Configuration file config.json not found!")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Configuration file format error: {e}")
            return {}
    
    def display_welcome(self):
        """Display welcome information"""
        print("\n" + "="*60)
        print("ICD-11 OPRO Prompt Optimization System")
        print("   Intelligent prompt optimization based on Optimization by PROmpting")
        print("="*60)
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check system status
        self._check_system_status()
        print()
    
    def _check_system_status(self):
        """Check system status"""
        print("\nSystem status check:")
        
        # Check seed prompts
        seed_dir = "opro/seed_prompts"
        if os.path.exists(seed_dir):
            seed_files = [f for f in os.listdir(seed_dir) if f.endswith('.txt')]
            print(f"   [OK] Seed prompts: {len(seed_files)} files")
        else:
            print("   [ERROR] Seed prompts directory not found")
        
        # Check test cases
        test_file = "tests/test_cases.json"
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                test_count = len(test_data.get('test_cases', []))
            print(f"   [OK] Test cases: {test_count} cases")
        else:
            print("   [ERROR] Test cases file not found")
        
        # Check configuration
        if self.config:
            print("   [OK] Configuration file loaded successfully")
        else:
            print("   [ERROR] Configuration file loading failed")
        
        # Check API key
        api_settings = self.config.get('api_settings', {})
        if api_settings.get('openai_api_key'):
            print("   [OK] OpenAI API key configured")
        else:
            print("   [WARNING] OpenAI API key not configured")
        
        # Check output directory
        if not os.path.exists("prompts"):
            os.makedirs("prompts")
            print("   [CREATED] Output directory: prompts/")
        else:
            print("   [OK] Output directory exists")
    
    def display_main_menu(self):
        """Display main menu"""
        print("\nPlease select operation mode:")
        print("   1. Run OPRO optimization - Automatically optimize prompts")
        print("   2. Evaluate existing prompts - Test prompt performance")
        print("   3. View optimization history - Monitor optimization progress")
        print("   4. Compare prompt variants - Compare different versions")
        print("   5. System configuration management - Adjust optimization parameters")
        print("   6. Batch processing mode - Process multiple prompts")
        print("   0. Exit system")
        print("-" * 50)
    
    def get_user_choice(self) -> str:
        """Get user choice"""
        while True:
            try:
                choice = input("Please enter option number (0-6): ").strip()
                if choice in ['0', '1', '2', '3', '4', '5', '6']:
                    return choice
                else:
                    print("[ERROR] Invalid option, please enter a number between 0-6")
            except KeyboardInterrupt:
                print("\n\nThanks for using ICD-11 OPRO system!")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] Input error: {e}")
    
    def run_opro_optimization(self):
        """Run OPRO optimization"""
        print("\n🚀 Starting OPRO optimization process...")
        
        try:
            # Display optimization parameters
            opro_settings = self.config.get('opro_settings', {})
            print(f"📊 Optimization parameters:")
            print(f"   Max iterations: {opro_settings.get('max_iterations', 20)}")
            print(f"   Improvement threshold: {opro_settings.get('improvement_threshold', 0.05)}")
            print(f"   Meta model: {opro_settings.get('meta_llm_model', 'gpt-4')}")
            
            # Confirm start
            confirm = input("\nStart optimization? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("❌ Optimization cancelled")
                return
            
            # Initialize optimizer
            if not self.optimizer:
                self.optimizer = OPROOptimizer()
            
            # Run optimization
            print("\n⏳ Running OPRO optimization, please wait...")
            result = self.optimizer.optimize_prompts()
            
            # Display results
            self._display_optimization_result(result)
            
        except Exception as e:
            logger.error(f"Error occurred during optimization: {e}")
            print(f"❌ Optimization failed: {e}")
    
    def _display_optimization_result(self, result: OptimizationResult):
        """Display optimization results"""
        print("\n" + "="*50)
        print("🎉 OPRO optimization completed!")
        print("="*50)
        print(f"📈 Final score: {result.final_score:.3f}")
        print(f"📊 Total improvement: {result.improvement_achieved:.3f}")
        print(f"🔄 Total iterations: {result.total_iterations}")
        print(f"⏱️  Time elapsed: {result.time_elapsed:.1f} seconds")
        
        if result.best_prompt.evaluation_details:
            print(f"\n📋 Detailed scores:")
            for dimension, score in result.best_prompt.evaluation_details.items():
                print(f"   {dimension}: {score:.2f}/10")
        
        print(f"\n💾 Results saved to:")
        print(f"   📄 Optimized prompt: prompts/optimized_prompt.txt")
        print(f"   📊 Optimization history: prompts/optimization_history.json")
        
        # Ask whether to evaluate immediately
        evaluate_now = input("\nEvaluate optimized prompt immediately? (Y/n): ").strip().lower()
        if evaluate_now not in ['n', 'no']:
            self.evaluate_optimized_prompt()
    
    def evaluate_existing_prompt(self):
        """Evaluate existing prompts"""
        print("\n🔍 Prompt evaluation mode")
        
        # Choose evaluation target
        print("\nPlease select prompt to evaluate:")
        print("   1️⃣  Optimized prompt (prompts/optimized_prompt.txt)")
        print("   2️⃣  Seed prompts")
        print("   3️⃣  Custom file path")
        
        choice = input("Please choose (1-3): ").strip()
        
        try:
            if choice == '1':
                self.evaluate_optimized_prompt()
            elif choice == '2':
                self.evaluate_seed_prompts()
            elif choice == '3':
                self.evaluate_custom_prompt()
            else:
                print("❌ Invalid option")
        except Exception as e:
            logger.error(f"Error occurred during evaluation: {e}")
            print(f"❌ Evaluation failed: {e}")
    
    def evaluate_optimized_prompt(self):
        """Evaluate optimized prompt"""
        optimized_file = "prompts/optimized_prompt.txt"
        
        if not os.path.exists(optimized_file):
            print("❌ Optimized prompt file not found")
            print("   Please run OPRO optimization first")
            return
        
        print(f"\n🔍 Evaluating file: {optimized_file}")
        
        # Select evaluation method
        method = self._select_evaluation_method()
        
        # Execute evaluation
        result = evaluate_prompt_file(optimized_file, method)
        self._display_evaluation_result(result)
    
    def evaluate_seed_prompts(self):
        """Evaluate seed prompts"""
        seed_dir = "opro/seed_prompts"
        
        if not os.path.exists(seed_dir):
            print("❌ Seed prompts directory not found")
            return
        
        seed_files = [f for f in os.listdir(seed_dir) if f.endswith('.txt')]
        
        if not seed_files:
            print("❌ No seed prompt files found")
            return
        
        print(f"\n📋 Found {len(seed_files)} seed prompts:")
        for i, filename in enumerate(seed_files, 1):
            print(f"   {i}. {filename}")
        
        try:
            choice = int(input(f"Please select file to evaluate (1-{len(seed_files)}): ")) - 1
            if 0 <= choice < len(seed_files):
                selected_file = os.path.join(seed_dir, seed_files[choice])
                method = self._select_evaluation_method()
                result = evaluate_prompt_file(selected_file, method)
                self._display_evaluation_result(result)
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def evaluate_custom_prompt(self):
        """Evaluate custom prompt"""
        file_path = input("Please enter prompt file path: ").strip()
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        method = self._select_evaluation_method()
        result = evaluate_prompt_file(file_path, method)
        self._display_evaluation_result(result)
    
    def _select_evaluation_method(self) -> str:
        """Select evaluation method"""
        print("\n🔧 Please select evaluation method:")
        print("   1️⃣  Comprehensive evaluation (comprehensive) - Includes test cases")
        print("   2️⃣  Fast evaluation (fast) - Based on heuristic rules")
        print("   3️⃣  LLM evaluation (llm_based) - Use large model evaluation")
        
        method_map = {
            '1': 'comprehensive',
            '2': 'fast', 
            '3': 'llm_based'
        }
        
        choice = input("Please choose (1-3): ").strip()
        return method_map.get(choice, 'fast')
    
    def _display_evaluation_result(self, result: EvaluationResult):
        """Display evaluation results"""
        print("\n" + "="*50)
        print("📊 Evaluation Results")
        print("="*50)
        print(f"🎯 Overall score: {result.overall_score:.2f}/10")
        print(f"📅 Evaluation time: {result.timestamp}")
        print(f"🔧 Evaluation method: {result.evaluation_method}")
        
        print(f"\n📋 Dimension scores:")
        for dimension, score in result.dimension_scores.items():
            emoji = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
            print(f"   {emoji} {dimension.capitalize()}: {score:.2f}/10")
        
        if result.test_case_results:
            print(f"\n🧪 Test case results ({len(result.test_case_results)} cases):")
            for test_result in result.test_case_results:
                emoji = "✅" if test_result['score'] >= 7 else "⚠️" if test_result['score'] >= 5 else "❌"
                print(f"   {emoji} {test_result['test_case_id']}: {test_result['score']:.1f}/10 ({test_result['category']})")
        
        if result.feedback:
            print(f"\n💬 Detailed feedback:")
            print(f"   {result.feedback[:200]}...")
        
        # Save results
        if not self.evaluator:
            self.evaluator = PromptEvaluator()
        self.evaluator.save_evaluation_results([result])
        print(f"\n💾 Evaluation results saved")
    
    def view_optimization_history(self):
        """View optimization history"""
        history_file = "prompts/optimization_history.json"
        
        if not os.path.exists(history_file):
            print("❌ Optimization history file not found")
            print("   Please run OPRO optimization first")
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            print("\n" + "="*50)
            print("📊 Optimization History Overview")
            print("="*50)
            print(f"📅 Completion time: {history_data.get('optimization_completed', 'N/A')}")
            print(f"🎯 Final score: {history_data.get('final_score', 0):.3f}")
            print(f"📈 Total improvement: {history_data.get('improvement_achieved', 0):.3f}")
            print(f"🔄 Total iterations: {history_data.get('total_iterations', 0)}")
            print(f"⏱️  Time elapsed: {history_data.get('time_elapsed', 0):.1f} seconds")
            
            candidates = history_data.get('candidates', [])
            if candidates:
                print(f"\n📋 Candidate prompt history ({len(candidates)} items):")
                for i, candidate in enumerate(candidates[-10:], 1):  # Show last 10 only
                    print(f"   {i:2d}. Score: {candidate.get('score', 0):.3f} | "
                          f"Iteration: {candidate.get('iteration', 0):2d} | "
                          f"Method: {candidate.get('generation_method', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Failed to read optimization history: {e}")
            print(f"❌ Unable to read optimization history: {e}")
    
    def compare_prompt_variants(self):
        """Compare prompt variants"""
        print("\n🔄 Prompt comparison mode")
        print("   This function will compare the performance of multiple prompts")
        
        # Collect files to compare
        files_to_compare = []
        
        # Check optimized prompt
        optimized_file = "prompts/optimized_prompt.txt"
        if os.path.exists(optimized_file):
            files_to_compare.append(("Optimized prompt", optimized_file))
        
        # Check seed prompts
        seed_dir = "opro/seed_prompts"
        if os.path.exists(seed_dir):
            seed_files = [f for f in os.listdir(seed_dir) if f.endswith('.txt')]
            for seed_file in seed_files:
                files_to_compare.append((f"Seed:{seed_file}", os.path.join(seed_dir, seed_file)))
        
        if len(files_to_compare) < 2:
            print("❌ At least two prompt files are needed for comparison")
            return
        
        print(f"\n📋 Found {len(files_to_compare)} comparable files:")
        for i, (name, _) in enumerate(files_to_compare, 1):
            print(f"   {i}. {name}")
        
        # Execute batch evaluation
        try:
            if not self.evaluator:
                self.evaluator = PromptEvaluator()
            
            prompts = []
            names = []
            for name, filepath in files_to_compare:
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompts.append(f.read())
                    names.append(name)
            
            print("\n⏳ Evaluating all prompts...")
            results = self.evaluator.batch_evaluate_prompts(prompts, method="fast")
            
            # Display comparison results
            self._display_comparison_results(names, results)
            
        except Exception as e:
            logger.error(f"Error occurred during comparison: {e}")
            print(f"❌ Comparison failed: {e}")
    
    def _display_comparison_results(self, names, results):
        """Display comparison results"""
        print("\n" + "="*70)
        print("🔄 Prompt Comparison Results")
        print("="*70)
        
        # Sort by total score
        sorted_results = sorted(zip(names, results), key=lambda x: x[1].overall_score, reverse=True)
        
        print(f"{'Rank':<4} {'Name':<25} {'Total':<8} {'Relevance':<8} {'Empathy':<8} {'Accuracy':<8} {'Safety':<8}")
        print("-" * 70)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            scores = result.dimension_scores
            print(f"{rank:<4} {name:<25} {result.overall_score:<8.2f} "
                  f"{scores.get('relevance', 0):<8.1f} "
                  f"{scores.get('empathy', 0):<8.1f} "
                  f"{scores.get('accuracy', 0):<8.1f} "
                  f"{scores.get('safety', 0):<8.1f}")
        
        # Display best performance
        best_name, best_result = sorted_results[0]
        print(f"\n🏆 Best performance: {best_name}")
        print(f"   Total score: {best_result.overall_score:.2f}/10")
    
    def manage_system_config(self):
        """System configuration management"""
        print("\n⚙️ System configuration management")
        print("   1️⃣  View current configuration")
        print("   2️⃣  Modify API settings")
        print("   3️⃣  Adjust optimization parameters")
        print("   4️⃣  Modify evaluation weights")
        
        choice = input("Please choose (1-4): ").strip()
        
        if choice == '1':
            self._display_current_config()
        elif choice == '2':
            self._modify_api_settings()
        elif choice == '3':
            self._modify_optimization_params()
        elif choice == '4':
            self._modify_evaluation_weights()
        else:
            print("❌ Invalid option")
    
    def _display_current_config(self):
        """Display current configuration"""
        print("\n📋 Current system configuration:")
        print(json.dumps(self.config, indent=2, ensure_ascii=False))
    
    def _modify_api_settings(self):
        """Modify API settings"""
        print("\n🔑 API settings modification")
        print("   Note: API keys will be displayed as *** format")
        
        api_settings = self.config.get('api_settings', {})
        
        # OpenAI API key
        current_openai = "***Set***" if api_settings.get('openai_api_key') else "Not set"
        print(f"\nCurrent OpenAI API key: {current_openai}")
        new_openai = input("Enter new OpenAI API key (leave blank to keep unchanged): ").strip()
        if new_openai:
            api_settings['openai_api_key'] = new_openai
            print("✅ OpenAI API key updated")
        
        # Anthropic API key
        current_anthropic = "***Set***" if api_settings.get('anthropic_api_key') else "Not set"
        print(f"\nCurrent Anthropic API key: {current_anthropic}")
        new_anthropic = input("Enter new Anthropic API key (leave blank to keep unchanged): ").strip()
        if new_anthropic:
            api_settings['anthropic_api_key'] = new_anthropic
            print("✅ Anthropic API key updated")
        
        # Save configuration
        self.config['api_settings'] = api_settings
        self._save_config()
    
    def _modify_optimization_params(self):
        """Modify optimization parameters"""
        print("\n🔧 Optimization parameter adjustment")
        
        opro_settings = self.config.get('opro_settings', {})
        
        # Maximum iterations
        current_max_iter = opro_settings.get('max_iterations', 20)
        print(f"\nCurrent maximum iterations: {current_max_iter}")
        new_max_iter = input("Enter new maximum iterations (leave blank to keep unchanged): ").strip()
        if new_max_iter:
            try:
                opro_settings['max_iterations'] = int(new_max_iter)
                print("✅ Maximum iterations updated")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Improvement threshold
        current_threshold = opro_settings.get('improvement_threshold', 0.05)
        print(f"\nCurrent improvement threshold: {current_threshold}")
        new_threshold = input("Enter new improvement threshold (leave blank to keep unchanged): ").strip()
        if new_threshold:
            try:
                opro_settings['improvement_threshold'] = float(new_threshold)
                print("✅ Improvement threshold updated")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Save configuration
        self.config['opro_settings'] = opro_settings
        self._save_config()
    
    def _modify_evaluation_weights(self):
        """Modify evaluation weights"""
        print("\n⚖️ Evaluation weight adjustment")
        print("   Weight sum should be 1.0")
        
        weights = self.config.get('evaluation', {}).get('weights', {})
        
        print(f"\nCurrent weights:")
        for dimension, weight in weights.items():
            print(f"   {dimension}: {weight}")
        
        print(f"\nPlease enter new weight values:")
        new_weights = {}
        total_weight = 0
        
        for dimension in ['relevance', 'empathy', 'accuracy', 'safety']:
            current = weights.get(dimension, 0.25)
            new_value = input(f"{dimension} ({current}): ").strip()
            if new_value:
                try:
                    new_weights[dimension] = float(new_value)
                    total_weight += float(new_value)
                except ValueError:
                    print(f"❌ {dimension} using original value")
                    new_weights[dimension] = current
                    total_weight += current
            else:
                new_weights[dimension] = current
                total_weight += current
        
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠️ Weight sum is {total_weight:.3f}, will be automatically normalized to 1.0")
            for dimension in new_weights:
                new_weights[dimension] /= total_weight
        
        # Save configuration
        if 'evaluation' not in self.config:
            self.config['evaluation'] = {}
        self.config['evaluation']['weights'] = new_weights
        self._save_config()
        print("✅ Evaluation weights updated")
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print("💾 Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            print(f"❌ Failed to save configuration: {e}")
    
    def batch_processing_mode(self):
        """Batch processing mode"""
        print("\n📦 Batch processing mode")
        print("   This mode can process multiple prompt files")
        
        # Select processing directory
        directory = input("Please enter directory path containing prompt files: ").strip()
        if not os.path.exists(directory):
            print(f"❌ Directory does not exist: {directory}")
            return
        
        # Scan files
        txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        if not txt_files:
            print("❌ No .txt files found in directory")
            return
        
        print(f"\n📋 Found {len(txt_files)} files:")
        for i, filename in enumerate(txt_files[:10], 1):  # Show first 10 only
            print(f"   {i}. {filename}")
        if len(txt_files) > 10:
            print(f"   ... and {len(txt_files) - 10} more files")
        
        # Confirm processing
        confirm = input(f"\nProcess all {len(txt_files)} files? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Batch processing cancelled")
            return
        
        # Execute batch processing
        try:
            if not self.evaluator:
                self.evaluator = PromptEvaluator()
            
            print("\n⏳ Processing batch...")
            prompts = []
            filenames = []
            
            for filename in txt_files:
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompts.append(f.read())
                    filenames.append(filename)
            
            results = self.evaluator.batch_evaluate_prompts(prompts, method="fast")
            
            # Display result summary
            print(f"\n📊 Batch processing completed!")
            print(f"Total files: {len(results)}")
            
            scores = [r.overall_score for r in results]
            print(f"Average score: {sum(scores) / len(scores):.2f}")
            print(f"Highest score: {max(scores):.2f}")
            print(f"Lowest score: {min(scores):.2f}")
            
            # Save results
            self.evaluator.save_evaluation_results(results, "batch_evaluation_results.json")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            print(f"❌ Batch processing failed: {e}")
    
    def run(self):
        """Run main program"""
        self.display_welcome()
        
        while True:
            self.display_main_menu()
            choice = self.get_user_choice()
            
            if choice == '0':
                print("\n👋 Thanks for using ICD-11 OPRO system!")
                break
            elif choice == '1':
                self.run_opro_optimization()
            elif choice == '2':
                self.evaluate_existing_prompt()
            elif choice == '3':
                self.view_optimization_history()
            elif choice == '4':
                self.compare_prompt_variants()
            elif choice == '5':
                self.manage_system_config()
            elif choice == '6':
                self.batch_processing_mode()
            
            # Pause for user to view results
            input("\nPress Enter to continue...")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ICD-11 OPRO Prompt Optimization System')
    parser.add_argument('--mode', choices=['optimize', 'evaluate', 'compare'], 
                       help='Directly specify run mode')
    parser.add_argument('--prompt-file', help='Path to prompt file to evaluate')
    parser.add_argument('--evaluation-method', choices=['comprehensive', 'fast', 'llm_based'],
                       default='fast', help='Evaluation method')
    
    args = parser.parse_args()
    
    # If command line arguments are specified, execute corresponding operations directly
    if args.mode:
        if args.mode == 'optimize':
            optimizer = OPROOptimizer()
            result = optimizer.optimize_prompts()
            print(f"Optimization completed! Final score: {result.final_score:.3f}")
        
        elif args.mode == 'evaluate':
            if not args.prompt_file:
                print("Error: Evaluation mode requires --prompt-file parameter")
                sys.exit(1)
            
            result = evaluate_prompt_file(args.prompt_file, args.evaluation_method)
            print(f"Evaluation completed! Total score: {result.overall_score:.2f}")
            for dim, score in result.dimension_scores.items():
                print(f"{dim}: {score:.2f}")
        
        elif args.mode == 'compare':
            interface = OPROInterface()
            interface.compare_prompt_variants()
    
    else:
        # Run interactive interface
        interface = OPROInterface()
        interface.run()

if __name__ == "__main__":
    main() 