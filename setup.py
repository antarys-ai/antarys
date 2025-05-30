from antarys import __version__
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="antarys",
    version=__version__,
    url="https://github.com/antarys-ai/antarys-python/",
    description="High-performance Python client for Antarys vector database",
    long_description=description,
    long_description_content_type="text/markdown",
    author="Antarys",
    author_email="antarys.ai@gmail.com",
    packages=find_packages(),
    package_data={"antarys": ["py.typed"]},
    install_requires=[
        "httpx>=0.25.0",
        "numpy>=1.24.0",
        "orjson>=3.9.0",
        "tqdm>=4.65.0",
        "cachetools>=5.3.0"
    ],
    extras_require={
        "all": [
            "numba>=0.58.0",
            "lz4>=4.3.2",
            "ujson>=5.8.0"
        ],
        "acceleration": [
            "numba>=0.58.0"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
    ],
)
