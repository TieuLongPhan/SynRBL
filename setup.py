from setuptools import setup, find_packages

setup(
    name='SynRBL',
    version='1.0.0',
    description='Synthesis Rebalancing Framework for Computational Chemistry',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/SynRBL',
    packages=find_packages(),
    install_requires=[
        'rdkit', 
        'networkx', 
        'pysmiles', 
        'tmap', 
        'map4'
        # add other dependencies as needed
    ],
    python_requires='>=3.9',
    # Additional metadata like classifiers, keywords, etc.
)
