from setuptools import setup, find_packages

setup(
    name="pystacknet",
    version="0.1",
    description="Python wrapper for StackNet model",
    author="German Novikov",
    author_email="german.novikov@phystech.edu",
    packages=find_packages(include=["pystacknet*"]),
    install_requires=["numpy>=1.14.0",
                      "pandas>=0.22.0",
                      "scipy>=1.0.0"]
)
