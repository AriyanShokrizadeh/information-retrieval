from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    include_package_data=True,
    author="Ariyan",
    version="0.0.1",
    name="ir",
    description="Information Retrieval Project",
    author_email="ariyanshokrizadeh@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
)
