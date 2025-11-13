from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="clusteringgo",
    version="0.1.0",
    author="Yonch",
    author_email="yehonatan.levin@mail.huji.ac.il",
    description="A package for analyzing gene expression clusters based on Gene Ontology.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<your_package_repository_url>", #TODO
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'run_clusteringgo=run_analysis:main',
        ],
    },
)
