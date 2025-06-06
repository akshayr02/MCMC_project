from setuptools import setup

setup(
    name='emulator_3pt',
    version='0.1.0',
    description='Code for emulating 2x3pt statistics',
    url='',
    author='Pierre Burger',
    author_email='pierre.burger@uwaterloo.ca',
    packages=['emulator_3pt'],
    install_requires=['numpy',
                      'scipy',
                      'jax',
                      'jaxlib',
                      'blackjax',
                      'gdown',
                      'matplotlib',
                      'nautilus-sampler',
                      'datetime',
                      'getdist'
                    ],
    python_requires=">=3.10",  # or your supported version
)