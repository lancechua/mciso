import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mciso",
    author="Lance Chua",
    author_email="lancerobinchua@gmail.com",
    description="Monte Carlo Inventory Stocking Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lancechua/mciso",
    project_urls={
        "Bug Tracker": "https://github.com/lancechua/mciso/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
