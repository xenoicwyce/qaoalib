from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

exec(open('qaoalib/version.py').read()) # puts __version__ into the namespace

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
        "numpy>=1.17",
        "matplotlib",
        "networkx",
        "qiskit>=0.25.0",
        "pydantic",
    ]
)
