import subprocess
import sys
from importlib import import_module
from typing import Dict


LIBRARIES: Dict[str, str] = {
    "pandas": "pandas",
    "openpyxl": "openpyxl",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "transformers": "transformers",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "numpy": "numpy",
    "streamlit": "streamlit",
    "reportlab": "reportlab",
    "tqdm": "tqdm",
}


def install_libraries() -> None:
    """
    Install required Python libraries using the current Python executable.
    """
    print(f"Using Python executable: {sys.executable}")
    for pip_name in LIBRARIES.keys():
        print(f"\nInstalling {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"{pip_name} installed successfully.")
        except subprocess.CalledProcessError as exc:
            print(f"Failed to install {pip_name}: {exc}")


def print_versions() -> None:
    """
    Import each library and print its version to confirm installation.
    """
    print("\n=== Library Versions ===")
    for pip_name, module_name in LIBRARIES.items():
        try:
            module = import_module(module_name)
            version = getattr(module, "__version__", None)
            if version is None:
                try:
                    import importlib.metadata as metadata
                except ImportError:
                    import importlib_metadata as metadata  # type: ignore
                version = metadata.version(module_name)
            print(f"{module_name}: {version}")
        except Exception as exc:  # pragma: no cover - best effort reporting
            print(f"Could not determine version for {module_name}: {exc}")


def main() -> None:
    """
    Entry point for installing libraries and printing their versions.
    """
    install_libraries()
    print_versions()


if __name__ == "__main__":
    main()

