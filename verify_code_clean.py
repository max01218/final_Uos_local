#!/usr/bin/env python3
"""
Code Verification Script
Verify that all code files contain no Chinese characters, emojis, or special symbols
"""

import os
import re
from pathlib import Path

def check_file_for_non_ascii(file_path):
    """Check if file contains non-ASCII characters"""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for non-ASCII characters
        non_ascii_chars = re.findall(r'[^\x00-\x7F]', content)
        if non_ascii_chars:
            # Get unique characters
            unique_chars = list(set(non_ascii_chars))
            issues.append(f"Non-ASCII characters found: {unique_chars}")
            
        # Check for common emojis
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]'
        emojis = re.findall(emoji_pattern, content)
        if emojis:
            unique_emojis = list(set(emojis))
            issues.append(f"Emojis found: {unique_emojis}")
            
        # Check for specific problematic symbols
        problematic_symbols = ['✅', '❌', '🎉', '📊', '🔧', '⚠️', '🆘', '✓']
        found_symbols = [symbol for symbol in problematic_symbols if symbol in content]
        if found_symbols:
            issues.append(f"Problematic symbols found: {found_symbols}")
            
    except Exception as e:
        issues.append(f"Error reading file: {e}")
        
    return issues

def verify_code_cleanliness():
    """Verify all Python files are clean"""
    
    print("Code Cleanliness Verification")
    print("=" * 50)
    print()
    
    # Define files and directories to check
    files_to_check = [
        "fastapi_server.py",
        "generate_faiss_index.py", 
        "process_icd11_data.py"
    ]
    
    dirs_to_check = [
        "CBT_System",
        "ICD11_OPRO"
    ]
    
    total_issues = 0
    clean_files = 0
    
    # Check specific files
    print("Checking specific files:")
    print("-" * 30)
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            issues = check_file_for_non_ascii(file_path)
            if issues:
                print(f"ISSUES in {file_path}:")
                for issue in issues:
                    print(f"  - {issue}")
                total_issues += len(issues)
            else:
                print(f"CLEAN: {file_path}")
                clean_files += 1
        else:
            print(f"NOT FOUND: {file_path}")
    
    print()
    
    # Check directories
    print("Checking directories:")
    print("-" * 30)
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"Checking {dir_path}/:")
            
            # Find all Python files in directory
            py_files = list(Path(dir_path).rglob("*.py"))
            
            for py_file in py_files:
                rel_path = os.path.relpath(py_file)
                issues = check_file_for_non_ascii(py_file)
                
                if issues:
                    print(f"  ISSUES in {rel_path}:")
                    for issue in issues:
                        print(f"    - {issue}")
                    total_issues += len(issues)
                else:
                    print(f"  CLEAN: {rel_path}")
                    clean_files += 1
        else:
            print(f"NOT FOUND: {dir_path}")
    
    print()
    print("=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Clean files: {clean_files}")
    print(f"Total issues found: {total_issues}")
    print()
    
    if total_issues == 0:
        print("SUCCESS: All checked files are clean!")
        print("No Chinese characters, emojis, or special symbols found.")
        return True
    else:
        print("WARNING: Issues found in some files.")
        print("Please fix the reported issues.")
        return False

def main():
    """Main function"""
    success = verify_code_cleanliness()
    
    if success:
        print()
        print("Your codebase is ready for production!")
        print("All files comply with ASCII-only requirements.")
    else:
        print()
        print("Please review and fix the issues above.")
        
    return success

if __name__ == "__main__":
    main() 