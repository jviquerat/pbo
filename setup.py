from setuptools import setup

setup(
    name='pbo',
    version='0.0.1',
    entry_points = {
        'console_scripts': ['pbo=pbo.src.main:main'],
    }
)
