"""Project configuration loader for audio concatenation projects."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def get_projects_dir() -> Path:
    """Get the projects directory path."""
    return Path(__file__).parent.parent.parent.parent / "projects"


def list_projects() -> List[str]:
    """List all available project names."""
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return []

    return sorted([
        p.stem for p in projects_dir.glob("*.yaml")
        if p.stem != "README"
    ])


def load_project(name: str) -> Dict:
    """Load a project configuration by name.

    Args:
        name: Project name (without .yaml extension)

    Returns:
        Project configuration dictionary

    Raises:
        FileNotFoundError: If project config doesn't exist
    """
    projects_dir = get_projects_dir()
    config_path = projects_dir / f"{name}.yaml"

    if not config_path.exists():
        available = list_projects()
        raise FileNotFoundError(
            f"Project '{name}' not found. "
            f"Available projects: {', '.join(available) or 'none'}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ["name", "path", "sample_prefix"]
    missing = [f for f in required if f not in config]
    if missing:
        raise ValueError(f"Project config missing required fields: {missing}")

    # Convert path to Path object and validate
    config["path"] = Path(config["path"])
    if not config["path"].exists():
        raise FileNotFoundError(f"Project path does not exist: {config['path']}")

    # Set defaults
    config.setdefault("sample_pattern", "{prefix}_{num}_")
    config.setdefault("num_samples", 50)
    config.setdefault("transpositions", [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4])
    config.setdefault("crossfade_ms", 1)
    config.setdefault("output_prefix", config["name"])
    config.setdefault("generate_midi", False)

    return config


def get_scales_dir(config: Dict) -> Path:
    """Get the scales_dir path for a project."""
    return config["path"] / "scales_dir"


def get_output_dir(config: Dict) -> Path:
    """Get or create the output directory for a project."""
    output_dir = config["path"] / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def format_sample_name(config: Dict, num: int) -> str:
    """Format a sample name according to the project's pattern.

    Args:
        config: Project configuration
        num: Sample/measure number

    Returns:
        Formatted sample name pattern for matching
    """
    pattern = config["sample_pattern"]
    prefix = config["sample_prefix"]

    # Handle different format patterns
    if "{num:04d}" in pattern:
        return pattern.format(prefix=prefix, num=num)
    elif "{num}" in pattern:
        return pattern.format(prefix=prefix, num=num)
    else:
        # Simple pattern like "prefix_num_"
        return f"{prefix}_{num}_"
