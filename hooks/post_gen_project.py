import shutil
import subprocess
from pathlib import Path
from typing import List

SERVICE: str = "{{ cookiecutter.services }}"

RAG_ONLY_DIRS: list[str] = [
    "src/domain/embedding",
    "src/domain/vector",
    "src/infrastructure/embedding",
    "src/infrastructure/vector",
    "tests/infrastructure/embedding",
    "tests/infrastructure/vector",
]


def remove_rag_components() -> None:
    """Remove embedding and vector DB directories when rag is not selected."""
    if SERVICE == "rag":
        return
    print("Service is not 'rag'. Removing embedding and vector DB components...")
    for dir_path in RAG_ONLY_DIRS:
        target = Path(dir_path)
        if target.exists():
            shutil.rmtree(target)


def remove_empty_python_files() -> None:
    """Remove empty .py files (0 bytes or only whitespace) from root."""
    print("Removing empty files...")
    for path in Path(".").rglob("*.py"):
        if path.is_file() and not path.read_text().strip():
            path.unlink()


def remove_empty_directories() -> None:
    """Remove empty directories left after Jinja2 conditional rendering."""
    print("Removing empty directories...")
    for path in sorted(Path(".").rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

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
    remove_rag_components()
    remove_empty_python_files()
    remove_empty_directories()
    format_project_code()
    run_template_tests()
    print("Template has been created successfully! :)")