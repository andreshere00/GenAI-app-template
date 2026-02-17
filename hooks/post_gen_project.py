import subprocess
import sys
from pathlib import Path
from typing import List

def remove_empty_python_files() -> None:
    """Remove empty .py files (0 bytes or only whitespace) from root."""
    print("Removing empty files...")
    for path in Path(".").rglob("*.py"):
        if path.is_file() and not path.read_text().strip():
            path.unlink()

def run_uv_command(args: list[str]) -> None:
    """Execute a command using uv run"""
    try:
        subprocess.run(["uv", "run", *args], check=False, capture_output=True)
    except FileNotFoundError:
        print(f"Error: 'uv' not found. Could not run {' '.join(args)}.")

def run_template_tests() -> None:
    """Execute tests located in the test/ directory."""
    test_path = Path("tests")
    if test_path.exists():
        print("Running initial tests...")
        run_uv_command(["pytest", str(test_path)])

def format_project_code() -> None:
    """Format the generated project using black and ruff."""
    commands: List[List[str]] = [
    ["black", "."],
    ["ruff", "check", ".", "--fix"]
    ]
    for cmd in commands:
        try:
            print("Formatting code...")
            run_uv_command(cmd)
        except FileNotFoundError:
            print(f"Warning: {' '.join(cmd[:3])} not found. Skipping formatting.")

if __name__ == "__main__":
    remove_empty_python_files()
    format_project_code()
    run_template_tests()
    print("Template has been created successfully! :)")