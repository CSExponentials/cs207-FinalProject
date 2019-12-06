import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Exponentials-AD",
    version="0.0.1",
    author="Galit",
    author_email="glukin@mit.edu",
    description="Foward mode implemented.",
    long_description="supports vectors, operators, and various trig functions",
    long_description_content_type="text/markdown",
    url="https://github.com/CSExponentials/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
