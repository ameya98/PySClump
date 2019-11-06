import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysclump",
    version="0.0.2",
    author="Ameya Daigavane",
    author_email="ameya.d.98@gmail.com",
    description="SClump implemented in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ameya98/PySClump",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)