from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyturbseq',
    author='Aidan Winters',
    author_email='aidanfwinters@gmail.com',
    version='0.0.6',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['pyturbseq'],
    python_requires='>=3.8,<3.12',
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
        'hdbscan',
        'dcor',
    ]
)