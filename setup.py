from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("qaoalib/version.py", "r") as f:
    __version__ = f.read().strip()

with open("requirements.txt", "r")  as f:
    requirements = [package.rstrip() for package in f]

setup(
    name="qaoalib",
    version=__version__,
    author="Xinwei Lee",
    author_email="xenoicwyce@gmail.com",
    description="A package for QAOA Maxcut calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xenoicwyce/qaoalib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["test*"]),
    python_requires=">=3.9",
    install_requires=[
        "qiskit",
        "qiskit_optimization",
        "qiskit_aer",
        "numpy==1.26.0",
        "scipy",
        "pytest",
        "pytest-cov",
        "matplotlib",
        "networkx",
        "pydantic",
    ],
)
