<a href="https://github.com/Mathieumoalic/pyzfn/actions?query=workflow%3A%22Tests%22">
<img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/Mathieumoalic/pyzfn/Tests.svg">
</a>

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
