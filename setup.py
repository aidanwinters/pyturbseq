from setuptools import setup, find_packages

setup(
    name='pyturbseq',
    author='Aidan Winters',
    author_email='aidanfwinters@gmail.com',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>3.11',
    install_requires=[
        'scanpy',
        'scipy',
        'scikit-learn',
        'umap-learn',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'fastcluster',
        'adjusttext',
        'pydeseq2',
        'statsmodels==0.13.5',
        'adpbulk',
        'anndata',
    ]
)