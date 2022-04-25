from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qaoalib",
    version="0.1.1",
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
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "networkx",
        "qiskit>=0.25.0",
    ]
)