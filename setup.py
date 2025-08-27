from setuptools import setup, find_packages

setup(
    name='topokemp',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'snappy', 'torch', 'numpy', 'matplotlib', 'biopython', 'qutip', 'networkx',
        'rdkit', 'pyscf', 'pygame', 'chess', 'mido', 'midiutil', 'astropy', 'control',
        'pubchempy', 'dendropy', 'statsmodels', 'PuLP', 'sympy', 'mpmath', 'scipy',
        'pandas', 'tqdm', 'ecdsa'
    ],
    description='Unified Topological Knot-Embedding Meta-Processor',
    author='Grok',
    author_email='grok@example.com',
    url='https://github.com/yourusername/topokemp',
)
