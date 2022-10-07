import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "vgg19_extractor",
    version = "0.0.1",
    author = "author",
    author_email = "author@example.com",
    description = "short package description",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "package URL",
    project_urls = {
        "Bug Tracker": "package issues URL",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "vgg19_extractor"},
    packages = setuptools.find_packages(where="vgg19_extractor"),
    python_requires = ">=3.6"
)