from setuptools import setup, find_packages

setup(
    name='SynRBL',
    version='0.1.1',
    description='Synthesis Rebalancing Framework for Computational Chemistry',
    author='TieuLongPhan',
    author_email='ptlong8995@gmail.com',
    url='https://github.com/TieuLongPhan/SynRBL',
    packages=find_packages(),
    install_requires=[
        'rdkit==2023.9.5',
        'joblib'
        # add other dependencies as needed
    ],
    python_requires='>=3.11',
    # Additional metadata like classifiers, keywords, etc.
)
