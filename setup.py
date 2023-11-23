from setuptools import setup


setup(
    name='pyturbseq',
    author='Aidan Winters',
    author_email='aidanfwinters@gmail.com',
    version='0.1.0',
    # packages=['pyturbseq'],
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
        'statsmodels',
        'adpbulk',
        'anndata',
    ]
)