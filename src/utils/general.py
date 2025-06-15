from pathlib import Path
import time
import humanize
from typing import IO, Any, Callable, Dict, Optional, Union

import torch

try:  # for 1.8
    from pytorch_lightning.utilities.cloud_io import get_filesystem
except ImportError:  # for 1.9
    from pytorch_lightning.core.saving import get_filesystem

_PATH = Union[str, Path]
_DEVICE = Union[torch.device, str, int]
_MAP_LOCATION_TYPE = Optional[
    Union[_DEVICE, Callable[[_DEVICE], _DEVICE], Dict[_DEVICE, _DEVICE]]
]


class LogTime:
    def __init__(self, verbose=True, **humanize_kwargs) -> None:
        if "minimum_unit" not in humanize_kwargs.keys():
            humanize_kwargs["minimum_unit"] = 'microseconds'
        self.humanize_kwargs = humanize_kwargs
        self.elapsed = None
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """
        Exceptions are captured in *args, we’ll handle none, since failing can be timed anyway
        """
        self.elapsed = time.time() - self.start
        self.elapsed_str = humanize.precisedelta(self.elapsed, **self.humanize_kwargs)
        if self.verbose:
            print(f"Time Elapsed: {self.elapsed_str}")


def intersect_list(list1, list2):
    return list(set(list1).intersection(set(list2)))


def difference_list(list1, list2):
    return list(set(list1) - set(list2))


def union_list(list1, list2):
    return list(set(list1).union(set(list2)))


# Copied over pytorch_lightning.utilities.cloud_io.load as it was deprecated
def pl_load(
    path_or_url: Union[IO, _PATH],
    map_location: _MAP_LOCATION_TYPE = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.

    """
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(
            str(path_or_url),
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
        )
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)


def download_m4_data():
    import os
    import urllib.request

    # Create the folder if it doesn't exist
    folder_path = "data/m4"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Downloading M4-Hourly.csv
    url1 = "https://auto-arima-results.s3.amazonaws.com/M4-Hourly.csv"
    filename1 = "data/m4/M4-Hourly.csv"
    urllib.request.urlretrieve(url1, filename1)

    # Downloading M4-Hourly-test.csv
    url2 = "https://auto-arima-results.s3.amazonaws.com/M4-Hourly-test.csv"
    filename2 = "data/m4/M4-Hourly-test.csv"
    urllib.request.urlretrieve(url2, filename2)
    return filename1, filename2
