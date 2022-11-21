[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)

micromagnetic post processing library

Wrapper around a `zarr.hierarchy.Group` to implement convenience functions that work with the ouput of a modified mumax3.

## Installation

```
$ pip install llyr
```

## Usage

#### Creating

```python
import llyr
job = llyr.open("path/to/folder.zarr")
# or through any remote protocol
job = llyr.open("ssh://pcss:jobs/job1.zarr")
```
