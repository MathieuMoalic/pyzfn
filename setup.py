from setuptools import find_packages, setup

setup(
    name="pyzfn",
    version="0.1.0",
    description="micromagnetic post processing library",
    author="Mathieu Moalic",
    author_email="matmoa@pm.me",
    platforms=["any"],
    license="GPL-3.0",
    url="https://github.com/MathieuMoalic/pyzfn",
    packages=find_packages(),
    install_requires=[i.strip() for i in open("requirements.txt", "r").readlines()],
)
