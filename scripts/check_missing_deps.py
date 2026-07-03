# scripts/check_missing_deps.py
import ast
import os
import sys
from pathlib import Path

# Packages that are part of the standard library (Python 3.14+)
STD_LIB = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set()

# Mapping of import names to package names (if they differ)
IMPORT_TO_PKG = {
    "discord": "discord.py",
    "googleapiclient": "google-api-python-client",
    "yaml": "pyyaml",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "google_genai": "google-genai",
    "google": "google-auth", # Common root for google-auth, google-cloud, etc.
    "httpx": "httpx",
    "moonshine_onnx": "useful-moonshine-onnx",  # DP-238 voice STT (lazy-imported)
    # Add more as discovered
}

SKIP_DIRS = {"node_modules", "dist", ".git", "__pycache__"}

def get_all_imports(directory):
    imports = set()
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=str(path))
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.add(alias.name.split('.')[0])
                            elif isinstance(node, ast.ImportFrom):
                                if node.module and node.level == 0:
                                    imports.add(node.module.split('.')[0])
                except Exception as e:
                    print(f"Error parsing {path}: {e}")
    return imports

def get_requirements(req_file):
    if not os.path.exists(req_file):
        return set()
    
    reqs = set()
    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line or line.startswith("-r"):
                continue
            # Extract package name (ignoring versions)
            pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip().lower()
            reqs.add(pkg.replace("_", "-"))
    return reqs

def main():
    src_dir = "src"
    # DP-250: deps are split across requirements.in (lean base, used by CI) and
    # requirements-voice.in (heavy voice/STT stack, prod-only). An import is
    # satisfied if it appears in EITHER, so union both .in files here.
    req_files = ["requirements.in", "requirements-voice.in"]

    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} directory not found.")
        sys.exit(1)

    all_imports = get_all_imports(src_dir)
    requirements = set()
    for req_file in req_files:
        requirements |= get_requirements(req_file)
    
    # Filter out standard library and local imports
    missing = []
    for imp in all_imports:
        if imp in STD_LIB:
            continue
        if os.path.exists(os.path.join(src_dir, imp)) or os.path.exists(os.path.join(src_dir, f"{imp}.py")):
            continue
        if os.path.exists(imp) or os.path.exists(f"{imp}.py"): # Check project root
            continue
        
        # Check against package name mapping
        pkg_name = IMPORT_TO_PKG.get(imp, imp).lower().replace("_", "-")
        
        if pkg_name not in requirements:
            # Special case for sub-packages or common patterns
            if pkg_name != "src": # src is local
                missing.append((imp, pkg_name))

    if missing:
        print("\n[!] Error: Found imported packages missing from requirements.in / requirements-voice.in:")
        for imp, pkg in sorted(missing):
            print(f"    - Import '{imp}' requires package '{pkg}'")
        print("\nAdd these to requirements.in (base) or requirements-voice.in (voice/ML),"
              " then run scripts/sync_deps.ps1\n")
        sys.exit(1)
    else:
        print("\n[+] All imports are accounted for in requirements.in / requirements-voice.in\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
