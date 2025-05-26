#!/usr/bin/env python
"""
build_voxcraft.py  –  cross-platform helper

• If extern/voxcraft-sim is missing, clones it.
• Configures CMake headless (-DVOX_SDL=OFF -DVOX_OPENGL=OFF).
• Builds into build/voxcraft-bin/.
• Copies the resulting voxcraft-sim binary (or voxcraft-sim.exe)
  into src/xeno_ml/evolution/bin/.

Usage:
    poetry run python tools/build_voxcraft.py [--cpu] [--clean] [--debug]

Options:
    --cpu    Force CPU build even if CUDA is detected
    --clean  Clean previous build before starting
    --debug  Build in debug mode instead of release
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
ROOT = Path(__file__).resolve().parents[1]
EXTERN_DIR = ROOT / "extern" / "voxcraft-sim"
BUILD_DIR = ROOT / "build" / "voxcraft-bin"
BIN_DIR = ROOT / "src" / "xeno_ml" / "evolution" / "bin"
VOXCRAFT_REPO = "https://github.com/voxcraft/voxcraft-sim.git"
VOXCRAFT_VERSION = "main"  # Pin to specific version/tag if needed

def check_requirements() -> None:
    """Check if required build tools are available."""
    required_tools = {
        "cmake": ["cmake", "--version"],
        "git": ["git", "--version"],
    }
    
    for tool, cmd in required_tools.items():
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(f"Required tool '{tool}' not found. Please install it first.")

def has_cuda() -> bool:
    """Check if CUDA is available on the system."""
    try:
        subprocess.run(["nvcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def clean_build() -> None:
    """Clean previous build artifacts."""
    if BUILD_DIR.exists():
        logger.info("Cleaning previous build...")
        shutil.rmtree(BUILD_DIR)

def clone_repo() -> None:
    """Clone or update the voxcraft-sim repository."""
    if not EXTERN_DIR.exists():
        logger.info("Cloning voxcraft-sim...")
        EXTERN_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", VOXCRAFT_REPO, str(EXTERN_DIR)],
            check=True,
        )
    else:
        logger.info("Using existing extern/voxcraft-sim")
        # Update the repository
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=EXTERN_DIR,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", VOXCRAFT_VERSION],
            cwd=EXTERN_DIR,
            check=True,
        )

def configure_cmake(gpu: bool, build_type: str) -> None:
    """Configure CMake build."""
    cfg = [
        "cmake",
        "-S", str(EXTERN_DIR),
        "-B", str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DVOX_SDL=OFF",
        "-DVOX_OPENGL=OFF",
    ]
    
    if not gpu:
        cfg += ["-DVOX_CUDA=OFF"]
    
    if platform.system() == "Windows":
        cfg += ["-A", "x64"]  # 64-bit MSVC
    elif platform.system() == "Darwin":  # macOS
        cfg += ["-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64"]  # Universal binary
    
    logger.info(f"Configuring ({'GPU' if gpu else 'CPU'} build, {build_type} mode)...")
    subprocess.run(cfg, check=True)

def build_project() -> None:
    """Build the project using CMake."""
    logger.info("Building...")
    subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "--parallel"],
        check=True
    )

def copy_binary() -> None:
    """Copy the built binary to the target directory."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    exe_name = "voxcraft-sim" + (".exe" if platform.system() == "Windows" else "")
    built_bin = next((BUILD_DIR / "bin").rglob(exe_name))
    
    target_path = BIN_DIR / exe_name
    shutil.copy2(built_bin, target_path)
    logger.info(f"✓ Copied to {target_path}")

def main(force_cpu: bool = False, clean: bool = False, debug: bool = False) -> None:
    """Main build function."""
    try:
        check_requirements()
        
        if clean:
            clean_build()
        
        clone_repo()
        
        gpu = has_cuda() and not force_cpu and platform.system() != "Windows"
        build_type = "Debug" if debug else "Release"
        
        configure_cmake(gpu, build_type)
        build_project()
        copy_binary()
        
        logger.info("Build completed successfully!")
        
    except subprocess.CalledProcessError as exc:
        logger.error(f"Build failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
    except Exception as exc:
        logger.error(f"Build failed: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build helper for VoxCraft")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only build")
    parser.add_argument("--clean", action="store_true", help="Clean previous build")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    args = parser.parse_args()
    
    main(force_cpu=args.cpu, clean=args.clean, debug=args.debug) 