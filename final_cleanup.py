#!/usr/bin/env python
"""
Final Cleanup Script for Poker AI Project

This script performs maintenance tasks:
1. Removes __pycache__ directories
2. Removes redundant or temporary files
3. Optionally can self-delete after execution

Usage:
    python final_cleanup.py [--self-delete]
"""
import os
import shutil
import sys
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cleanup.log"),
        logging.StreamHandler()
    ]
)

# Define the project root directory
ROOT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

# Files that should be considered for cleanup (add patterns as needed)
FILES_TO_REMOVE = [
    # Specific files - already consolidated or backed up
    "actor_fixed.py",
    "all_in_one_poker_ai.py",
    "fixed_custom_beta.py",
    "fixed_map_functions.py",
    "fixed_showdown.py",
    "fixed_train_step.py",
    "integrate_optimizations.py", 
    "test_optimizations.py",
    "consolidation_plan.md",
    "OPTIMIZATION_GUIDE.md",
    "temp_custom_beta.py",
    "cleanup_project.py",
    "fix_script.py",
    
    # Pattern-based files - extend as needed
    "*.pyc",
    "*.tmp",
    "*.bak",
    "*_temp.py",
    "temp_*.py",
]

# Cleanup __pycache__ directories
def cleanup_pycache(directory):
    """Remove all __pycache__ directories recursively."""
    count = 0
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                try:
                    logging.info(f"Removing {pycache_path}")
                    shutil.rmtree(pycache_path)
                    count += 1
                except PermissionError:
                    logging.warning(f"Permission error when trying to delete {pycache_path}")
                except Exception as e:
                    logging.error(f"Error removing {pycache_path}: {e}")
    return count

def remove_files():
    """Remove specified files from the project."""
    count = 0
    # Remove exact files
    for file in FILES_TO_REMOVE:
        if not ('*' in file):
            file_path = ROOT_DIR / file
            if file_path.exists():
                try:
                    logging.info(f"Removing {file_path}")
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logging.error(f"Error removing {file_path}: {e}")
    
    # Remove pattern-based files
    for pattern in [p for p in FILES_TO_REMOVE if '*' in p]:
        for filepath in ROOT_DIR.glob(pattern):
            if filepath.is_file():
                try:
                    logging.info(f"Removing {filepath}")
                    filepath.unlink()
                    count += 1
                except Exception as e:
                    logging.error(f"Error removing {filepath}: {e}")
    
    return count

def main():
    """Main cleanup function."""
    logging.info("Starting Poker AI project cleanup...")
    
    # Remove files that are no longer needed
    file_count = remove_files()
    logging.info(f"Removed {file_count} files")
    
    # Cleanup __pycache__ directories
    pycache_count = cleanup_pycache(ROOT_DIR)
    logging.info(f"Removed {pycache_count} __pycache__ directories")
    
    logging.info("Cleanup complete!")

if __name__ == "__main__":
    main()
    
    # For self-deletion of this script
    if "--self-delete" in sys.argv:
        script_path = os.path.abspath(__file__)
        logging.info("Self-deletion requested. Script will be removed.")
        # Use a separate process to delete this script after execution
        if os.name == 'nt':  # Windows
            os.system(f'ping -n 2 127.0.0.1 > nul && del "{script_path}"')
        else:  # Unix/Linux
            os.system(f'sleep 1 && rm "{script_path}" &')
            
    # Additional help information
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nCleaning options:")
        print("  --self-delete  : Delete this script after execution")
        print("  --help, -h     : Show this help message")
        sys.exit(0)
