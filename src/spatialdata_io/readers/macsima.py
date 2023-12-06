from __future__ import annotations

from collections.abc import Mapping
from math import e
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger

from spatialdata_io._constants._constants import MacsimaKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._utils import parse_physical_size, calc_scale_factors, add_ome_attr

from spatialdata._logging import logger
import spatialdata as sd
from ome_types import from_tiff
import dask.array as da
from aicsimageio import AICSImage

__all__ = ["macsima"]


@inject_docs(vx=MacsimaKeys)
def macsima(
    path: str | Path,
    metadata: bool = False,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    subset: int | None = None,
    c_subset: int | None = None,
    max_chunk_size: int = 1024,
    c_chunks_size: int = 1,
    multiscale: bool = True,
    transformations: bool = None,
    scale_factors: list[int] | None = None,
    write_ome = True,
    default_scale_factor: int = 2,
    exposure_time: float | str = 'highest',
    output_path: Path | None = None,
) -> SpatialData:
    """
    Read *MACSima* formatted dataset.

    This function reads images from a MACSima cyclic imaging experiment. Metadata of the cycles is parsed from the image names.

    .. seealso::

        - `MACSima output <https://application.qitissue.com/getting-started/naming-your-datasets>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    metadata
        Whether to search for a .txt file with metadata in the folder. If False, the metadata in the image names is used.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    subset
        Subset the image to the first ``subset`` pixels in x and y dimensions.
    c_subset
        Subset the image to the first ``c_subset`` channels.
    max_chunk_size
        Maximum chunk size for x and y dimensions.
    c_chunks_size
        Chunk size for c dimension.
    multiscale
        Whether to create a multiscale image.
    transformations
        Whether to add a transformation from pixels to microns to the image.
    scale_factors
        Scale factors to use for downsampling. If None, scale factors are calculated based on image size.
    default_scale_factor
        Default scale factor to use for downsampling.
    exposure_time
        Exposure time to use for images with multiple exposure times. Can be 'highest' or a float.
    output_path
        Path to write the :class:`spatialdata.SpatialData` object to. If None, the object is not written.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    if '*' in str(path):
        paths = list(expandpath(path, filter_suffix=['.zarr']))
        logger.info(f"Expanded path to: {paths}")
    else:
        path = Path(path)
        paths = [path] + [x for x in path.iterdir() if x.is_dir()]
    # check all paths to make sure they exist
    for p in paths:
        if not p.exists():
            logger.warning(f"Cannot find path: {p}")
    images = {p.stem: get_image(
        p,
        metadata=metadata,
        imread_kwargs=imread_kwargs,
        subset=subset,
        c_subset=c_subset,
        max_chunk_size=max_chunk_size,
        c_chunks_size=c_chunks_size,
        multiscale=multiscale,
        transformations=transformations,
        scale_factors=scale_factors,
        write_ome=write_ome,
        default_scale_factor=default_scale_factor,
        exposure_time=exposure_time,
        coordinate_system=p.stem,
    ) for p in paths}
    # filter out None values and raise warning
    for k, v in images.copy().items():
        if v is None:
            logger.warning(f"Removing {k} as it is None")
            images.pop(k)
    sdata = sd.SpatialData(images=images, table=None)
    if output_path is not None:
        sdata.write(output_path, overwrite=True)
    return sdata

    
def get_image(
    path: str | Path,
    metadata: bool = False,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    subset: int | None = None,
    c_subset: int | None = None,
    max_chunk_size: int = 1024,
    c_chunks_size: int = 1,
    multiscale: bool = True,
    transformations: bool = None,
    scale_factors: list[int] | None = None,
    write_ome = True,
    exposure_time: float | str = 'highest',
    default_scale_factor: int = 2,
    coordinate_system: str = "global",
):
    path = Path(path)
    if transformations is None:
        transformations = write_ome
    path_files = []
    pixels_to_microns = None
    # if metadata:
    #     # read metadata to get list of images and channel names
    #     path_files = list(path.glob(f'*{MacsimaKeys.METADATA_SUFFIX}'))
    #     if len(path_files) > 0:
    #         if len(path_files) > 1: 
    #             logger.warning(f"Cannot determine metadata file. Expecting a single file with format .txt. Got multiple files: {path_files}")
    #         path_metadata = list(path.glob(f'*{MacsimaKeys.METADATA_SUFFIX}'))[0]
    #         df = pd.read_csv(path_metadata, sep="\t", header=0, index_col=None)
    #         logger.debug(df)
    #         df['channel'] = df['ch1'].str.split(' ').str[0]
    #         df['round_channel'] = df['Round'] + ' ' + df['channel']
    #         path_files = [path / p for p in df.filename.values]
    #         assert all([p.exists() for p in path_files]), f"Cannot find all images in metadata file. Missing: {[p for p in path_files if not p.exists()]}"
    #         round_channels = df.round_channel.values
    #         stack, sorted_channels = get_stack(path_files, round_channels, imread_kwargs)
    #     else:
    #         logger.warning(f"Cannot find metadata file. Will try to parse from image names.")
    if not metadata or len(path_files) == 0:
        # get list of image paths, get channel name from OME data and cycle number from filename
        # look for OME-TIFF files
        ome_patt = f'*{MacsimaKeys.IMAGE_OMETIF}*'
        path_files = list(path.glob(ome_patt))
        # sort based on name
        path_files = natural_sort(path_files)
        if not path_files:
            # look for .qptiff files
            qptif_patt = f'*{MacsimaKeys.IMAGE_QPTIF}*'
            path_files = list(path.glob(qptif_patt))
            logger.debug(path_files)
            if not path_files:
                logger.warning(f"Cannot determine files for {path}. Expecting '{ome_patt}' or '{qptif_patt}' files")
                return None
            # TODO: warning if not 1 ROI with 1 .qptiff per cycle
            # TODO: robuster parsing of {name}_cycle{round}_{scan}.qptiff
            rounds = [f"R{int(p.stem.split('_')[1][5:])}" for p in path_files]
            # parse .qptiff files
            imgs = [AICSImage(img, **imread_kwargs) for img in path_files]
            # sort based on cycle number
            rounds, imgs = zip(*sorted(zip(rounds, imgs), key=lambda x: int(x[0][1:])))
            channels_per_round = [img.channel_names for img in imgs]
            # take first image and first channel to get physical size
            ome_data = imgs[0].ome_metadata
            logger.debug(ome_data)
            pixels_to_microns = parse_physical_size(ome_pixels=ome_data.images[0].pixels)
            da_per_round = [img.dask_data[0, :, 0, :, :] for img in imgs]
            sorted_channels = []
            for r, cs in zip(rounds, channels_per_round):
                for c in cs:
                    sorted_channels.append(f"{r} {c}")
            stack = da.stack(da_per_round).squeeze()
            # Parse OME XML
            # img.ome_metadata
            # arr = img.dask_data[0, :, 0, :, :]
            # channel_names = img.channel_names
            logger.debug(sorted_channels)
            logger.debug(stack)
        else:
            logger.debug(path_files)
            # make sure not to remove round 0 when parsing!
            try:
                # 001_S_R-01_W-B-1_ROI-01_A-CD14REA599ROI1_C-REA599.ome.tif
                rounds = [f"R{int(p.stem.split('_')[0])}" for p in path_files]
            except ValueError:
                try:
                    # R-1_W-B-1_G-1_C-0_autofluorescence_FITCVNone.FITC_75.0_FINAL.ome.tif
                    # R-2_W-C-1_G-1_C-0_autofluorescence_FITCVNone.FITC_75.0_FINAL.ome.tif
                    rounds = [f"R{int(p.stem.split('C-')[-1].split('_')[0])}" for p in path_files]
                except ValueError:
                    # fallback to just using numbers
                    rounds = [str(i) for i in range(len(path_files))]
            channels = [from_tiff(p).images[0].pixels.channels[0].name for p in path_files]
            # if channels is all None, use channel names from file name
            if all([c is None for c in channels]):
                channels = [p.stem.split('C-')[-1].split('_')[1] for p in path_files]
            round_channels = [f"{r} {c}" for r, c in zip(rounds, channels)]
            stack, sorted_channels = get_stack(path_files, round_channels, imread_kwargs)
            # if any channel names are the same, presume multiple exposure time
            if len(sorted_channels) != len(set(sorted_channels)):
                if exposure_time == 'highest':
                    # use last exposure time
                    logger.debug(sorted_channels)
                    # go through the channels and stack in reverse order and only output the first occurence of each unique channel
                    duplicate_sorted_channels = sorted_channels
                    sorted_channels = []
                    stack_index = []
                    for i, c in enumerate(duplicate_sorted_channels[::-1]):
                        if c not in sorted_channels:
                            stack_index.append(len(duplicate_sorted_channels) - i - 1)
                            sorted_channels.append(c)
                    logger.debug(sorted_channels)
                    # reverse again to get correct order
                    sorted_channels = sorted_channels[::-1]
                    stack = stack[stack_index[::-1], :, :]
                else:
                    raise NotImplementedError(f"Exposure time {exposure_time} not implemented")

    
    # do subsetting if needed
    if subset:
        stack = stack[:, :subset, :subset]
    if c_subset:
        stack = stack[:c_subset, :, :]
        sorted_channels = sorted_channels[:c_subset]
    if multiscale and not scale_factors:
        scale_factors = calc_scale_factors(stack, default_scale_factor=default_scale_factor)
    if not multiscale:
        scale_factors = None
    logger.debug(f"Scale factors: {scale_factors}")

    t_dict = None
    if transformations:
        pixels_to_microns = pixels_to_microns or parse_physical_size(path_files[0])
        t_pixels_to_microns = sd.transformations.Scale([pixels_to_microns, pixels_to_microns], axes=("x", "y"))
        # 'microns' is also used in merscope example
        # no inverse needed as the transformation is already from pixels to microns
        t_dict = {
            coordinate_system: t_pixels_to_microns,
        }
    # # chunk_size can be 1 for channels
    chunks = {
        "x": max_chunk_size,
        "y": max_chunk_size,
        "c": c_chunks_size,
    }
    stack = sd.models.Image2DModel.parse(
        stack,
        # TODO: make sure y and x locations are correct
        dims=["c", "y", "x"],
        scale_factors=scale_factors,
        chunks=chunks,
        c_coords=sorted_channels,
        transformations=t_dict if transformations else None,
    )
    return stack

def get_stack(path_files: list[Path], round_channels: list[str], imread_kwargs: Mapping[str, Any], sort=False)-> Any:
    imgs_channels = list(zip(path_files, round_channels))
    logger.debug(imgs_channels)
    # sort based on round number
    if sort:
        imgs_channels = sorted(imgs_channels, key=lambda x: int(''.join([d for d in x[1].split(' ')[0] if d.isdigit()])))
    logger.debug(f'Len imgs_channels: {len(imgs_channels)}')
    # read in images and merge channels
    sorted_paths, sorted_channels = list(zip(*imgs_channels))
    imgs = [imread(img, **imread_kwargs) for img in sorted_paths]
    stack = da.stack(imgs).squeeze()
    return stack, sorted_channels

def expandpath(path_pattern, filter_suffix=None, only_dir=True) -> Iterable[Path]:
    p = Path(path_pattern).expanduser()
    parts = p.parts[p.is_absolute():]
    output = Path(p.root).glob(str(Path(*parts)))
    if only_dir:
        output = [p for p in output if p.is_dir()]
    if filter_suffix:
        output = [p for p in output if p.suffix not in filter_suffix]
    return output

def natural_sort(l): 
    # Could also use natsort package
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    return sorted(l, key=alphanum_key)