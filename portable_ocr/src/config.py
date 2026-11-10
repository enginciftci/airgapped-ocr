"""Portable Air-Gapped Configuration, DLL Preloader, and Relative Path Resolution."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

# Enforce 100% air-gapped offline execution before any third-party imports
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Base portable directory: <portable_ocr>/
SRC_DIR: Path = Path(__file__).resolve().parent
PORTABLE_ROOT: Path = SRC_DIR.parent

RUNTIME_DIR: Path = PORTABLE_ROOT / "runtime"
PYTHON_DIR: Path = RUNTIME_DIR / "python"
LLAMA_DIR: Path = RUNTIME_DIR / "llama"

MODELS_DIR: Path = PORTABLE_ROOT / "models"
DOCLING_MODELS_DIR: Path = MODELS_DIR / "docling"
VLM_MODELS_DIR: Path = MODELS_DIR / "vlm"

DEFAULT_INPUT_DIR: Path = PORTABLE_ROOT / "input"
DEFAULT_OUTPUT_DIR: Path = PORTABLE_ROOT / "output"


def _preload_portable_vc_redist() -> None:
    """Register portable DLL directories and preload MSVC runtime DLLs into memory.

    Once loaded by absolute path into the current process, Windows reuses the loaded
    module handles whenever torch, onnxruntime, opencv, or scipy request them, even
    when C:\\Windows\\System32 lacks the Visual C++ Redistributable.
    """
    if sys.platform != "win32":
        return

    site_pkgs = PYTHON_DIR / "Lib" / "site-packages"
    raw_dirs = [
        PYTHON_DIR,
        PYTHON_DIR / "DLLs",
        site_pkgs / "torch" / "lib",
        site_pkgs / "onnxruntime" / "capi",
        site_pkgs / "cv2",
        site_pkgs / "pymupdf",
        LLAMA_DIR,
    ]
    dll_dirs: list[Path] = []
    for d in raw_dirs:
        if d.is_dir():
            try:
                rp = Path(os.path.realpath(str(d)))
                if rp.is_dir() and rp not in dll_dirs:
                    dll_dirs.append(rp)
            except Exception:
                pass
            if d not in dll_dirs:
                dll_dirs.append(d)

    # Prepend to PATH for child processes and LoadLibraryW fallback
    existing_path = os.environ.get("PATH", "")
    prepend_str = os.pathsep.join(str(d) for d in dll_dirs)
    if prepend_str:
        os.environ["PATH"] = prepend_str + os.pathsep + existing_path

    # Register with Windows loader for Python 3.8+ LoadLibraryExW
    if hasattr(os, "add_dll_directory"):
        for d in dll_dirs:
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass

    # Preload VC++ Redistributable DLLs in dependency order
    vc_dll_order = [
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "vcruntime140_threads.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "concrt140.dll",
        "vccorlib140.dll",
        "vcomp140.dll",
    ]
    for dll_name in vc_dll_order:
        dll_path = PYTHON_DIR / dll_name
        if dll_path.is_file():
            try:
                ctypes.WinDLL(str(dll_path))
            except Exception:
                pass


_preload_portable_vc_redist()

SUPPORTED_IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
SUPPORTED_EXTENSIONS: set[str] = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}


def get_llama_server_exe() -> Path | None:
    """Return path to portable llama-server.exe if present."""
    candidate = LLAMA_DIR / "llama-server.exe"
    if candidate.is_file():
        return candidate
    for p in LLAMA_DIR.rglob("llama-server.exe"):
        return p
    return None


def get_vlm_model_paths() -> tuple[Path | None, Path | None]:
    """Return (model_gguf_path, mmproj_gguf_path) inside models/vlm."""
    if not VLM_MODELS_DIR.exists():
        return None, None
    ggufs = sorted(VLM_MODELS_DIR.glob("*.gguf"))
    model_path: Path | None = None
    mmproj_path: Path | None = None
    for f in ggufs:
        if "mmproj" in f.name.lower():
            mmproj_path = f
        else:
            model_path = f
    return model_path, mmproj_path
