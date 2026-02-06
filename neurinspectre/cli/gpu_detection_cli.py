#!/usr/bin/env python3
"""
GPU Detection CLI Module for NeurInSpectre
Provides comprehensive GPU detection and model inventory commands
"""

import sys
import json
import logging
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON to disk (best-effort create parent dirs)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _framework_versions() -> Dict[str, str]:
    """Best-effort inventory of common AI framework versions."""
    out: Dict[str, str] = {}
    for mod in ("torch", "transformers", "huggingface_hub", "tensorflow", "onnxruntime"):
        try:
            __import__(mod)
            m = sys.modules.get(mod)
            out[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            continue
    return out


def _scan_hf_cache(limit: int = 25) -> List[Dict[str, Any]]:
    """Scan local HuggingFace cache for cached models (no network)."""
    hits: List[Dict[str, Any]] = []
    # Preferred: huggingface_hub cache scanner
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if getattr(repo, "repo_type", None) != "model":
                continue
            rid = str(getattr(repo, "repo_id", ""))
            size = int(getattr(repo, "size_on_disk", 0) or 0)
            hits.append(
                {
                    "model_id": rid,
                    "size_bytes": size,
                    "size_mb": float(size) / (1024 * 1024) if size else 0.0,
                    "source": "huggingface_hub.scan_cache_dir",
                }
            )
        hits.sort(key=lambda d: float(d.get("size_bytes", 0)), reverse=True)
        return hits[: max(1, int(limit))]
    except Exception:
        pass

    # Fallback: ~/.cache/huggingface/hub
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.exists():
            return []
        for model_dir in cache_dir.iterdir():
            if not (model_dir.is_dir() and model_dir.name.startswith("models--")):
                continue
            body = model_dir.name.replace("models--", "")
            parts = body.split("--")
            model_id = "/".join(parts) if parts else body

            total = 0
            try:
                for p in model_dir.rglob("*"):
                    if p.is_file():
                        total += p.stat().st_size
            except Exception:
                total = 0
            if total <= 0:
                continue

            hits.append(
                {
                    "model_id": model_id,
                    "size_bytes": total,
                    "size_mb": float(total) / (1024 * 1024),
                    "source": "manual_scan",
                }
            )
        hits.sort(key=lambda d: float(d.get("size_bytes", 0)), reverse=True)
        return hits[: max(1, int(limit))]
    except Exception:
        return []


def _scan_model_files(base: Path, *, limit: int = 25) -> List[Dict[str, Any]]:
    """Scan for local model artifacts under a base path (best-effort)."""
    exts = (
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".h5",
        ".hdf5",
        ".keras",
        ".tflite",
        ".joblib",
        ".pkl",
    )
    out: List[Dict[str, Any]] = []
    try:
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            try:
                size = int(p.stat().st_size)
            except Exception:
                continue
            if size < 128 * 1024:
                continue
            out.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": size,
                    "size_mb": float(size) / (1024 * 1024),
                }
            )
    except Exception:
        return []
    out.sort(key=lambda d: float(d.get("size_bytes", 0)), reverse=True)
    return out[: max(1, int(limit))]

def run_gpu_detection_command(args) -> int:
    """Main entry point for GPU detection commands"""
    try:
        if args.gpu_command == 'detect':
            return run_universal_detection(args)
        elif args.gpu_command == 'models':
            return run_model_inventory(args)
        elif args.gpu_command == 'monitor':
            return run_gpu_monitoring(args)
        elif args.gpu_command == 'nvidia':
            return run_nvidia_specific(args)
        elif args.gpu_command == 'apple':
            return run_apple_specific(args)
        else:
            logger.error(f"Unknown GPU command: {args.gpu_command}")
            return 1
    except Exception as e:
        logger.error(f"GPU detection command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def run_universal_detection(args) -> int:
    """Run universal GPU detection across all platforms"""
    try:
        # Import the universal detector from the installed package (no repo-root dependency)
        from ..universal_gpu_detector import UniversalGPUDetector
        
        print("🌍 NeurInSpectre Universal GPU Detection")
        print("=" * 45)
        
        detector = UniversalGPUDetector()
        summary = detector.run_full_detection()
        
        # Save report if requested (write directly to provided path)
        if args.output:
            output_path = Path(args.output)
            try:
                _write_json(output_path, detector.last_report or summary)
                print(f"\n💾 Report saved to: {output_path}")
            except Exception as save_err:
                logger.error(f"Could not save GPU report to {output_path}: {save_err}")
        
        # Print summary
        print("\n🎉 Detection Summary:")
        if summary['detected_gpus']:
            print(f"   ✅ GPU Types: {', '.join(summary['detected_gpus'])}")
            print(f"   🔧 Frameworks: {', '.join(summary['supported_frameworks'])}")
        else:
            print("   ❌ No GPU acceleration detected")
        
        # Integration with NeurInSpectre
        print("\n🧠 NeurInSpectre Integration:")
        if 'Apple Silicon (MPS)' in summary['detected_gpus']:
            print("   ✅ NeurInSpectre MPS support: Available")
            print("   🚀 Recommended: Use --device mps for optimal performance")
        elif any('NVIDIA' in gpu for gpu in summary['detected_gpus']):
            print("   ✅ NeurInSpectre CUDA support: Available")
            print("   🚀 Recommended: Use --device cuda for optimal performance")
        else:
            print("   ⚠️  NeurInSpectre will use CPU mode")
            print("   💡 Consider upgrading to GPU-enabled hardware for better performance")
        
        return 0
        
    except SyntaxError as e:
        # Typically indicates a corrupted/stale install (e.g., partial overwrite left a `from __future__`
        # import mid-file). Provide a concrete remediation path.
        loc = f"{getattr(e, 'filename', 'unknown')}:{getattr(e, 'lineno', '?')}"
        msg = getattr(e, "msg", None) or str(e)
        logger.error(f"Universal GPU detector import failed (syntax error at {loc}): {msg}")
        logger.info("Fix: upgrade/reinstall NeurInSpectre in this environment (e.g., `pip install -U -e .` from repo root).")
        return 1
    except ImportError as e:
        logger.error(f"Universal GPU detector not available: {e}")
        logger.info("This command requires the packaged universal GPU detector module.")
        return 1
    except Exception as e:
        logger.error(f"Universal detection failed: {e}")
        return 1

def run_model_inventory(args) -> int:
    """Run AI model inventory scan"""
    try:
        quick = bool(getattr(args, "quick", False))
        print("📦 NeurInSpectre Model Inventory")
        print("=" * 35)
        print(f"💻 Platform: {platform.system()} {platform.machine()}")

        frameworks = _framework_versions()
        if frameworks:
            fw_str = ", ".join([f"{k}={v}" for k, v in sorted(frameworks.items())])
            print(f"🔧 Frameworks: {fw_str}")
        else:
            print("🔧 Frameworks: (none detected)")

        hf_models = _scan_hf_cache(limit=15 if quick else 50)
        cwd_models = _scan_model_files(Path.cwd(), limit=15 if quick else 50)

        print(f"\n🤗 HuggingFace cached models: {len(hf_models)}")
        for m in hf_models[:5]:
            print(f"   - {m.get('model_id')} ({m.get('size_mb', 0.0):.1f} MB)")
        if len(hf_models) > 5:
            print(f"   ... and {len(hf_models) - 5} more")

        print(f"\n📄 Model files under current directory: {len(cwd_models)}")
        for m in cwd_models[:5]:
            print(f"   - {m.get('name')} ({m.get('size_mb', 0.0):.1f} MB)")
        if len(cwd_models) > 5:
            print(f"   ... and {len(cwd_models) - 5} more")

        payload = {
            "timestamp": datetime.now().isoformat(),
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "frameworks": frameworks,
            "huggingface_cached_models": hf_models,
            "cwd_model_files": cwd_models,
        }

        if getattr(args, "output", None):
            outp = Path(str(args.output))
            _write_json(outp, payload)
            print(f"\n💾 Report saved to: {outp}")

        return 0
        
    except Exception as e:
        logger.error(f"Model inventory failed: {e}")
        return 1

def run_gpu_monitoring(args) -> int:
    """Run GPU monitoring for running models"""
    try:
        if args.nvidia_only:
            print("⚡ NVIDIA GPU Model Monitoring")
            print("=" * 35)
            # Best-effort: rely on nvidia-smi if present
            def _print_nvidia_processes() -> None:
                try:
                    res = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    if res.returncode != 0:
                        print("❌ nvidia-smi not available on this system")
                        return
                    txt = (res.stdout or "").strip()
                    print(txt if txt else "(no running GPU processes)")
                except Exception as e:
                    print(f"❌ NVIDIA monitoring failed: {e}")

            _print_nvidia_processes()

            if getattr(args, "continuous", False):
                duration = int(getattr(args, "duration", 60) or 60)
                interval = int(getattr(args, "interval", 5) or 5)
                start = time.time()
                try:
                    while time.time() - start < duration:
                        time.sleep(interval)
                        print("\n⏱️  Update")
                        print("-" * 30)
                        _print_nvidia_processes()
                except KeyboardInterrupt:
                    print("\n⏹️  Monitoring stopped by user")
            
        else:
            # Universal monitoring
            from ..universal_gpu_detector import UniversalGPUDetector
            
            print("🌍 Universal GPU Monitoring")
            print("=" * 30)
            
            detector = UniversalGPUDetector()
            # Detector prints details itself (best-effort; may require psutil).
            detector.detect_running_models()
            
            if args.continuous:
                print(f"\n🔄 Continuous monitoring for {args.duration or 60} seconds...")
                start_time = time.time()
                duration = args.duration or 60
                interval = args.interval or 5
                
                try:
                    while time.time() - start_time < duration:
                        print(f"\n⏱️  Update at {time.strftime('%H:%M:%S')}")
                        print("-" * 30)
                        detector.detect_running_models()
                        time.sleep(interval)
                except KeyboardInterrupt:
                    print("\n⏹️  Monitoring stopped by user")
        
        return 0
        
    except ImportError as e:
        logger.error(f"GPU monitoring not available: {e}")
        return 1
    except Exception as e:
        logger.error(f"GPU monitoring failed: {e}")
        return 1

def run_nvidia_specific(args) -> int:
    """Run NVIDIA-specific detection and analysis"""
    try:
        print("⚡ NVIDIA GPU Analysis")
        print("=" * 30)
        try:
            res = subprocess.run(["nvidia-smi", "--version"], capture_output=True, text=True, timeout=5, check=False)
            if res.returncode != 0:
                print("⚠️  nvidia-smi not available on this system (skipping NVIDIA checks)")
                return 0
        except Exception:
            print("⚠️  nvidia-smi not available on this system (skipping NVIDIA checks)")
            return 0

        from ..universal_gpu_detector import UniversalGPUDetector

        detector = UniversalGPUDetector()
        detector.detect_nvidia()
        summary = detector.generate_summary()

        print(f"✅ Detected: {', '.join(summary.get('detected_gpus') or []) or 'none'}")
        fw = summary.get("supported_frameworks") or []
        if fw:
            print(f"🔧 Frameworks: {', '.join(fw)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"NVIDIA detection failed: {e}")
        return 1

def run_apple_specific(args) -> int:
    """Run Apple Silicon-specific detection and analysis"""
    try:
        if platform.machine() != 'arm64' or platform.system() != 'Darwin':
            print("❌ This is not an Apple Silicon system")
            return 1

        print("🍎 Apple Silicon Analysis")
        print("=" * 30)

        from ..universal_gpu_detector import UniversalGPUDetector

        detector = UniversalGPUDetector()
        mps_ok = detector.detect_apple_silicon()
        print(f"✅ PyTorch MPS available: {'YES' if mps_ok else 'NO'}")

        if not getattr(args, "skip_models", False):
            hf = _scan_hf_cache(limit=15 if getattr(args, "quick", False) else 50)
            print(f"🤗 HuggingFace cached models: {len(hf)}")
            for m in hf[:5]:
                print(f"   - {m.get('model_id')} ({m.get('size_mb', 0.0):.1f} MB)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Apple Silicon analysis failed: {e}")
        return 1

def add_gpu_commands(subparsers):
    """Add GPU detection commands to the CLI parser"""
    
    # Main GPU command group
    gpu_parser = subparsers.add_parser(
        'gpu', 
        help='🎮 GPU detection and model inventory tools',
        description='Comprehensive GPU detection, model inventory, and monitoring tools'
    )
    gpu_subparsers = gpu_parser.add_subparsers(dest='gpu_command', help='GPU detection commands')
    
    # Universal detection command
    detect_parser = gpu_subparsers.add_parser(
        'detect', 
        help='🌍 Universal GPU detection (Mac Silicon, NVIDIA, AMD, Intel)'
    )
    detect_parser.add_argument('--output', '-o', help='Output report file (JSON)')
    detect_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Model inventory command
    models_parser = gpu_subparsers.add_parser(
        'models', 
        help='📦 AI model inventory and analysis'
    )
    models_parser.add_argument('--quick', '-q', action='store_true', help='Quick scan (faster)')
    models_parser.add_argument('--output', '-o', help='Output report file (JSON)')
    models_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # GPU monitoring command
    monitor_parser = gpu_subparsers.add_parser(
        'monitor', 
        help='🔄 Monitor running AI models on GPU'
    )
    monitor_parser.add_argument('--nvidia-only', action='store_true', help='NVIDIA-specific monitoring')
    monitor_parser.add_argument('--continuous', '-c', action='store_true', help='Continuous monitoring')
    monitor_parser.add_argument('--duration', '-d', type=int, default=60, help='Monitoring duration (seconds)')
    monitor_parser.add_argument('--interval', '-i', type=int, default=5, help='Update interval (seconds)')
    monitor_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # NVIDIA-specific command
    nvidia_parser = gpu_subparsers.add_parser(
        'nvidia', 
        help='⚡ NVIDIA GPU detection and analysis'
    )
    nvidia_parser.add_argument('--quick', '-q', action='store_true', help='Quick check (faster)')
    nvidia_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Apple Silicon-specific command
    apple_parser = gpu_subparsers.add_parser(
        'apple', 
        help='🍎 Apple Silicon GPU detection and analysis'
    )
    apple_parser.add_argument('--quick', '-q', action='store_true', help='Quick check (faster)')
    apple_parser.add_argument('--skip-models', action='store_true', help='Skip model inventory')
    apple_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return gpu_parser 