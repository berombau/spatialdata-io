from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import spatialdata as sd
import zarr
from anndata import AnnData, read_text
from h5py import File
from ome_types import from_tiff
from ome_types.model import Pixels, UnitsLength
from spatialdata._logging import logger

from spatialdata_io.readers._utils._read_10x_h5 import _read_10x_h5

PathLike = Union[os.PathLike, str]

try:
    from numpy.typing import NDArray

    NDArrayA = NDArray[Any]
except (ImportError, TypeError):
    NDArray = np.ndarray  # type: ignore[misc]
    NDArrayA = np.ndarray  # type: ignore[misc]


def _read_counts(
    path: str | Path,
    counts_file: str,
    library_id: Optional[str] = None,
    **kwargs: Any,
) -> tuple[AnnData, str]:
    path = Path(path)
    if counts_file.endswith(".h5"):
        adata: AnnData = _read_10x_h5(path / counts_file, **kwargs)
        with File(path / counts_file, mode="r") as f:
            attrs = dict(f.attrs)
            if library_id is None:
                try:
                    lid = attrs.pop("library_ids")[0]
                    library_id = lid.decode("utf-8") if isinstance(lid, bytes) else str(lid)
                except ValueError:
                    raise KeyError(
                        "Unable to extract library id from attributes. Please specify one explicitly."
                    ) from None

            adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
            for key in ["chemistry_description", "software_version"]:
                if key not in attrs:
                    continue
                metadata = attrs[key].decode("utf-8") if isinstance(attrs[key], bytes) else attrs[key]
                adata.uns["spatial"][library_id]["metadata"][key] = metadata

        return adata, library_id
    if library_id is None:
        raise ValueError("Please explicitly specify `library id`.")

    if counts_file.endswith((".csv", ".txt")):
        adata = read_text(path / counts_file, **kwargs)
    elif counts_file.endswith(".mtx.gz"):
        try:
            from scanpy.readwrite import read_10x_mtx
        except ImportError:
            raise ImportError("Please install scanpy to read 10x mtx files, `pip install scanpy`.")
        prefix = counts_file.replace("matrix.mtx.gz", "")
        adata = read_10x_mtx(path, prefix=prefix, **kwargs)
    else:
        raise NotImplementedError("TODO")

    adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
    return adata, library_id


def add_ome_attr(path: Path, pixels_to_microns: dict | None = None) -> None:
    """Add OME metadata to Zarr store of SpatialData object."""
    path = Path(path)
    assert path.exists()
    # store = zarr.open(path, mode="r")
    sdata = sd.SpatialData.read(path)
    logger.debug(sdata)
    # get subdirs
    images = [p for p in path.glob("images/*") if p.is_dir()]
    logger.debug(images)
    # write the image data
    for image in images:
        name = list(sdata.images.keys())[0]
        el = sdata.images[name]
        if "scale0" in el:
            el = el["scale0"]
            logger.debug("picking scale0")
        el = el[name]
        logger.debug(el)
        channel_names = list(el.coords["c"].data)
        logger.debug(channel_names)
        # get maximum possible value for dtype
        dtype = el.dtype
        max_value = np.iinfo(dtype).max
        logger.debug(f"{dtype} {max_value}")
        # store = parse_url(image, mode="w").store
        # list of 7 basic colors
        colors = ["ff0000", "00ff00", "0000ff", "ffff00", "00ffff", "ff00ff", "ffffff"]
        channel_dicts = [
            {
                "active": True if i < 3 else False,
                "label": c,
                "coefficient": 1,
                "family": "linear",
                "inverted": False,
                # get one of the 7 colors, based on i
                "color": colors[i % 7],
                "window": {
                    "min": 0,
                    "start": 0,
                    "end": max_value,
                    "max": max_value,
                },
            }
            for i, c in enumerate(channel_names)
        ]
        omero_dict = {
            "channels": channel_dicts,
            "rdefs": {
                "defaultT": 0,  # First timepoint to show the user
                "defaultZ": 0,  # First Z section to show the user
                "model": "color",  # "color" or "greyscale"
            },
            "name": name,
            "version": "0.4",
        }
        if pixels_to_microns:
            omero_dict["pixel_size"] = pixels_to_microns
        root = zarr.open_group(image)
        root.attrs.update({"omero": omero_dict})
    zarr.consolidate_metadata(path)


def calc_scale_factors(stack: Any, min_size: int = 1000, default_scale_factor: int = 2) -> list[int]:
    """Calculate scale factors based on image size to get lowest resolution under min_size pixels."""
    # get lowest dimension, ignoring channels
    lower_scale_limit = min(stack.shape[1:])
    scale_factor = default_scale_factor
    scale_factors = [scale_factor]
    lower_scale_limit /= scale_factor
    while lower_scale_limit >= min_size:
        # scale_factors are cumulative, so we don't need to do e.g. scale_factor *= 2
        scale_factors.append(scale_factor)
        lower_scale_limit /= scale_factor
    return scale_factors


def parse_channels(path: Path) -> list[str]:
    """Parse channel names from an OME-TIFF file."""
    images = from_tiff(path).images
    if len(images) > 1:
        logger.warning("Found multiple images in OME-TIFF file. Only the first one will be used.")
    channels = images[0].pixels.channels
    logger.debug(channels)
    names = [c.name for c in channels]
    return names


def parse_physical_size(path: Path | None = None, ome_pixels: Pixels | None = None) -> float:
    """Parse physical size from OME-TIFF to micrometer."""
    pixels = ome_pixels or from_tiff(path).images[0].pixels
    logger.debug(pixels)
    if pixels.physical_size_x_unit != pixels.physical_size_y_unit:
        logger.error("Physical units for x and y dimensions are not the same.")
        raise NotImplementedError
    if pixels.physical_size_x != pixels.physical_size_y:
        logger.error("Physical sizes for x and y dimensions are the same.")
        raise NotImplementedError
    # convert to micrometer if needed
    if pixels.physical_size_x_unit == UnitsLength.NANOMETER:
        physical_size = pixels.physical_size_x / 1000
    elif pixels.physical_size_x_unit == UnitsLength.MICROMETER:
        physical_size = pixels.physical_size_x
    else:
        logger.error(f"Physical unit not recognized: '{pixels.physical_size_x_unit}'.")
        raise NotImplementedError
    return float(physical_size)
