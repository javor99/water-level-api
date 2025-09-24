import importlib.util
import sys
from types import ModuleType

_config_module: ModuleType = None

def load_config(config_path: str = "config.py") -> dict:
    """Load or reload configuration from a Python file"""
    global _config_module
    
    # Always create new spec
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec from {config_path}")
    
    if _config_module is None:
        # First-time load
        _config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = _config_module
        spec.loader.exec_module(_config_module)
    else:
        # Hot reload - create new module instance
        new_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = new_module
        spec.loader.exec_module(new_module)
        _config_module = new_module
    
    return _config_module.CONFIG

def load_mappings(config_path: str = "mappings.py") -> dict:
    """Load or reload configuration from a Python file"""
    global _config_module
    
    # Always create new spec
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec from {config_path}")
    
    if _config_module is None:
        # First-time load
        _config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = _config_module
        spec.loader.exec_module(_config_module)
    else:
        # Hot reload - create new module instance
        new_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = new_module
        spec.loader.exec_module(new_module)
        _config_module = new_module
    
    return _config_module.STATION_MAPPING

def load_experiments(config_path: str = "experiments.py") -> dict:
    """Load or reload configuration from a Python file"""
    global _config_module
    
    # Always create new spec
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec from {config_path}")
    
    if _config_module is None:
        # First-time load
        _config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = _config_module
        spec.loader.exec_module(_config_module)
    else:
        # Hot reload - create new module instance
        new_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = new_module
        spec.loader.exec_module(new_module)
        _config_module = new_module
    
    return _config_module.EXPERIMENTS