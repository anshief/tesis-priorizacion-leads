import importlib
import pathlib


DEPENDENCIES = ("pandas", "numpy", "openpyxl")
REQUIRED_DIRS = (
    pathlib.Path("data/raw"),
    pathlib.Path("data/processed"),
    pathlib.Path("reports"),
    pathlib.Path("scripts"),
)


try:
    for dependency in DEPENDENCIES:
        importlib.import_module(dependency)
except ImportError as exc:
    print(f"ERROR: {exc}")
else:
    missing_dirs = [str(path) for path in REQUIRED_DIRS if not path.is_dir()]

    if missing_dirs:
        print(f"ERROR: faltan carpetas: {', '.join(missing_dirs)}")
    else:
        print("OK: dependencias y estructura correctas")
