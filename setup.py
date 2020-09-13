import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read()

requirements = requirements.split('\n')

setuptools.setup(
    name="ricebowl",
    version="0.4.3",
    author="Shivek Chhabra",
    author_email="shivekchhabra@gmail.com",
    description="This package allows the users to preprocess dataframes and images, plot the data and then use ml models using a single command",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shivekchhabra/ricebowl",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
