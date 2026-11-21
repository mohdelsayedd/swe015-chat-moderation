"""Environment and hardware report.

Why this file exists:
- Stage 1: proves the virtual environment installed correctly (every package imports).
- Stage 7: benchmark.py imports `hardware_summary()` so the results table states the
  hardware it was measured on (requirement R4: "state the hardware"). The hardware
  line comes from this script, not from memory.

Run from the repo root:
    python -m src.env_info
"""

import importlib
import os
import platform
import subprocess
import sys

# Packages from requirements.txt (import name, not pip name).
PACKAGES = [
    "torch", "transformers", "accelerate", "datasets", "huggingface_hub",
    "sklearn", "gensim", "scipy", "numpy", "pandas", "gradio",
]


def _cpu_name() -> str:
    """Return the CPU marketing name (e.g. 'Intel(R) Core(TM) i7-...').

    platform.processor() on Windows only gives 'Intel64 Family 6 Model ...',
    which is useless in a slide, so we read the real name per OS.
    """
    system = platform.system()
    try:
        if system == "Windows":
            import winreg  # standard library, Windows only
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            )
            if out.stdout.strip():
                return out.stdout.strip()
    except Exception:  # any failure -> fall back, never crash the report
        pass
    return platform.processor() or "unknown CPU"


def _ram_gb() -> str:
    """Total physical RAM in GB, using only the standard library."""
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return f"{stat.ullTotalPhys / 1024**3:.1f} GB"
        if hasattr(os, "sysconf"):  # Linux / macOS
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return f"{total / 1024**3:.1f} GB"
    except Exception:
        pass
    return "unknown"


def hardware_summary() -> dict:
    """Hardware facts used in the results-table caption (R4)."""
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram": _ram_gb(),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        info["torch_threads"] = torch.get_num_threads()
        info["torch_cuda_available"] = torch.cuda.is_available()
    except Exception as exc:  # ImportError, or OSError if Windows blocks torch's DLLs
        info["torch"] = f"could not load ({type(exc).__name__})"
    return info


def main() -> None:
    print("=== Hardware ===")
    for key, value in hardware_summary().items():
        print(f"{key:>22}: {value}")

    print("\n=== Packages ===")
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"{name:>22}: {getattr(module, '__version__', '?')}")
        except Exception as exc:  # show the real error, keep checking the rest
            ok = False
            print(f"{name:>22}: FAILED -> {type(exc).__name__}: {exc}")

    # Stage 6 relies on dynamic INT8 quantization. Check the CPU backend now,
    # so a problem shows up tonight and not at 3 a.m.
    print("\n=== Quantization check ===")
    try:
        import warnings

        import torch
        from torch import nn

        print("supported engines:", torch.backends.quantized.supported_engines)
        tiny = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
        with warnings.catch_warnings():
            # quantize_dynamic is deprecated upstream (moving to torchao) but
            # works in the pinned torch 2.15.0; we expect this warning.
            warnings.simplefilter("ignore", category=DeprecationWarning)
            from torch.ao.quantization import quantize_dynamic
            q = quantize_dynamic(tiny, {nn.Linear}, dtype=torch.qint8)
        out = q(torch.randn(1, 8))
        print("quantize_dynamic: OK  (output shape", tuple(out.shape), ")")
        print("quantized layer type:", type(q[0]).__module__ + "." + type(q[0]).__name__)
    except Exception as exc:
        ok = False
        print(f"quantize_dynamic: FAILED -> {type(exc).__name__}: {exc}")

    print("\nRESULT:", "ALL OK" if ok else "SOMETHING FAILED (see above)")


if __name__ == "__main__":
    main()
