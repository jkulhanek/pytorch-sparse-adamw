from setuptools import setup, find_packages
from torch_sparse_adamw import __version__

setup(
    name='torch_sparse_adamw',
    version=__version__,
    packages=find_packages(include=('torch_sparse_adamw', 'torch_sparse_adamw.*')),
    author='Jonáš Kulhánek',
    author_email='jonas.kulhanek@live.com',
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=[x.rstrip('\n') for x in open('requirements.txt')])
