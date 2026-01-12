from __future__ import annotations

from pathlib import Path


def list_tif_paths(data_base_dir: str, data_source: str) -> list[str]:
    """List .tif image paths for the requested data source.

    Args:
        data_base_dir (str): Root directory containing ``edente`` and/or ``dente`` subfolders.
        data_source (str): Either ``edente``, ``dente``, or ``both``.

    Returns:
        list[str]: Sorted list of .tif image paths.

    Raises:
        ValueError: If ``data_source`` is invalid.
        FileNotFoundError: If no .tif files are found.
    """
    base_path = Path(data_base_dir)
    direct_tifs = sorted(base_path.glob("*.tif"))
    if direct_tifs:
        return [str(path) for path in direct_tifs]

    if data_source == "edente":
        tif_paths = sorted((base_path / "edente").glob("*.tif"))
    elif data_source == "dente":
        tif_paths = sorted((base_path / "dente").glob("*.tif"))
    elif data_source == "both":
        tif_paths_edente = sorted((base_path / "edente").glob("*.tif"))
        tif_paths_dente = sorted((base_path / "dente").glob("*.tif"))
        tif_paths = tif_paths_edente + tif_paths_dente
    else:
        raise ValueError(f"data_source must be 'edente', 'dente', or 'both', got '{data_source}'")

    if len(tif_paths) == 0:
        raise FileNotFoundError(f"No .tif images found in {data_base_dir}/{data_source}")
    return [str(path) for path in tif_paths]
