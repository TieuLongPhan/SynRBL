from setuptools import setup, find_packages

setup(
    name='SynRBL',
    version='0.1.1',
    description='Synthesis Rebalancing Framework for Computational Chemistry',
    author='TieuLongPhan',
    author_email='your.email@example.com',
    url='https://github.com/TieuLongPhan/SynRBL',
    packages=find_packages(),
    install_requires=[
        'rdkit==2023.9.4',
        'joblib==1.3.2',
        'seaborn==0.13.2',
        'xgoost==2.0.3',
        'scikit_learn==1.4.1.post1',
        'imbalanced_learn==0.12.0',
        'reportlab==4.1.0'
 
    ],
    python_requires='>=3.11',
    # Additional metadata like classifiers, keywords, etc.
)






