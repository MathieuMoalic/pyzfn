[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)

![Tests](https://github.com/mathieumoalic/pyzfn/actions/workflows/tests.yml/badge.svg)

micromagnetic post processing library

Wrapper around `zarr.hierarchy.Group` from [zarr](https://zarr.readthedocs.io/en/stable/) to implement convenience functions that work with the ouput of a modified mumax3.

## Installation

```
pip install git+https://github.com/mathieumoalic/pyzfn
```

## Usage

#### Creating

```python
import llyr
job = llyr.open("path/to/folder.zarr")
# or through any remote protocol
job = llyr.open("ssh://pcss:jobs/job1.zarr")
```
