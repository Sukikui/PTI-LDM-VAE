from __future__ import annotations

from pathlib import Path


def resolve_input_relative_path(input_dir: str | Path) -> Path:
    """Resolve an input directory to a project-relative path segment.

    Args:
        input_dir (str | Path): Input directory path.

    Returns:
        Path: Relative path used for output structure.
    """
    input_path = Path(input_dir).resolve()
    try:
        return input_path.relative_to(Path.cwd())
    except ValueError:
        return Path(str(input_path).lstrip("/"))


def resolve_run_subdir(run_dir: str | Path, input_dir: str | Path, subdir: str) -> Path:
    """Build a run-relative output path for a given input directory.

    Args:
        run_dir (str | Path): Base run directory.
        input_dir (str | Path): Input directory to mirror.
        subdir (str): Subdirectory name (e.g., ``inference`` or ``eval``).

    Returns:
        Path: Resolved run subdirectory path.
    """
    relative_input = resolve_input_relative_path(input_dir)
    return Path(run_dir) / subdir / relative_input


def resolve_run_output_dir(
    run_dir: str | Path,
    input_dir: str | Path,
    output_dir: str | Path | None,
    subdir: str,
) -> Path:
    """Resolve the output directory, optionally overridden by the user.

    Args:
        run_dir (str | Path): Base run directory.
        input_dir (str | Path): Input directory to mirror.
        output_dir (str | Path | None): Optional override path.
        subdir (str): Subdirectory name (e.g., ``inference`` or ``eval``).

    Returns:
        Path: Output directory path.
    """
    if output_dir is not None:
        return Path(output_dir)
    return resolve_run_subdir(run_dir, input_dir, subdir)
