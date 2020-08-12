import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="ricebowl",
    version="1.0.0",
    author="Shivek Chhabra",
    author_email="shivekchhabra@gmail.com",
    description="This package allows the users to preprocess dataframes, images and then use ml models using a single command",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shivekchhabra/ricebowl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
