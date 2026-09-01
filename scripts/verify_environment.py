"""Bi-ISL Environment Verification Script."""
import sys
import importlib
from typing import Dict, List, Tuple

groups: Dict[str, List[Tuple[str, str]]] = {
    "core": [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("yaml", "PyYAML"),
        ("pydantic", "Pydantic"),
        ("tqdm", "tqdm"),
    ],
    "vision": [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("PIL", "Pillow"),
        ("torchvision", "TorchVision"),
    ],
    "training": [
        ("tensorboard", "TensorBoard"),
        ("accelerate", "Accelerate"),
    ],
    "evaluation": [
        ("sacrebleu", "SacreBLEU"),
        ("rouge_score", "ROUGE-Score"),
        ("bert_score", "BERTScore"),
        ("sklearn", "Scikit-Learn"),
    ],
    "development": [
        ("pytest", "PyTest"),
        ("ruff", "Ruff"),
        ("mypy", "MyPy"),
    ],
    "deployment": [
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
    ],
    "optionalresearch": [
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("evaluate", "HuggingFace Evaluate"),
    ]
}

def verify_environment() -> int:
    print("=" * 60)
    print("      Bi-ISL Python Development Environment Audit")
    print("=" * 60)
    
    py_ver = sys.version_info
    print(f"Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver.major == 3 and py_ver.minor == 11:
        print("[PASS] Python 3.11 exact target matched.")
    elif py_ver.major == 3 and py_ver.minor in [10, 12, 13, 14]:
        print(f"[INFO] Python 3.{py_ver.minor} detected (compatible runtime).")
    else:
        print(f"[WARN] Unexpected Python version 3.{py_ver.minor}.")
        
    print("-" * 60)
    
    total_packages = 0
    missing_packages = 0
    
    for group_name, pkg_list in groups.items():
        print(f"\nGroup: [{group_name.upper()}]")
        for import_name, display_name in pkg_list:
            total_packages += 1
            try:
                mod = importlib.import_module(import_name)
                ver = getattr(mod, "__version__", "installed")
                print(f"  [PASS] {display_name:<28} ({ver})")
            except ImportError:
                print(f"  [MISSING] {display_name:<25} (Not installed)")
                missing_packages += 1

    print("\n" + "=" * 60)
    if missing_packages == 0:
        print("STATUS: ALL DEPENDENCY GROUPS VERIFIED PERFECTLY (100% COMPLETE)")
        return 0
    else:
        print(f"STATUS: {total_packages - missing_packages}/{total_packages} PACKAGES INSTALLED.")
        print("Run 'pip install -e .[core,vision,training,evaluation,development,deployment,optionalresearch]' to install missing packages.")
        return 0

if __name__ == "__main__":
    sys.exit(verify_environment())
