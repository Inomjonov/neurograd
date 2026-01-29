from setuptools import setup, find_packages

setup(
    name="vectorgrad",                # package name on PyPI
    version="0.1.0",
    packages=find_packages(),         # automatically finds your package folder
    install_requires=[],              # add dependencies like numpy if needed
    python_requires=">=3.8",
    author="Mironshoh Inomjonov",
    author_email="mironshohinom@gmail.com",
    description="A micrograd-like autodiff library extended to vectors",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Inomjonov/vectorgrad",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
