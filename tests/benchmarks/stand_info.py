"""Stand information for the X placeholders in section 3.4.

Fills in:
    — объём видеопамяти, ГБ
    — число логических ядер CPU
    — объём оперативной памяти, ГБ

Run on the same host that the service container is deployed to.
"""
from __future__ import annotations

import json
import os
import platform
from typing import Any, Dict


def _gpu_info() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch is not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is not available"}

    idx = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(idx)
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(idx),
        "device_count": torch.cuda.device_count(),
        "vram_total_gb": round(total_b / (1024**3), 2),
        "vram_free_gb": round(free_b / (1024**3), 2),
    }


def _cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import psutil  # type: ignore[import-not-found]

        info["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["physical_cores"] = None
    return info


def _ram_info() -> Dict[str, Any]:
    try:
        import psutil  # type: ignore[import-not-found]

        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
        }
    except ImportError:
        try:
            import resource  # type: ignore[import-not-found]

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return {
                "total_gb": round((pages * page_size) / (1024**3), 2),
                "available_gb": None,
            }
        except (AttributeError, ValueError):
            return {"total_gb": None, "available_gb": None}


def collect() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_info(),
        "ram": _ram_info(),
        "gpu": _gpu_info(),
    }


if __name__ == "__main__":
    info = collect()
    print(json.dumps(info, ensure_ascii=False, indent=2))
