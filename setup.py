from setuptools import setup, find_packages

setup(
    name='DeepKT',
    version="0.3",
    author='lccasagrande',
    license="MIT",
    packages=find_packages(),
    python_requires='>=3.0',
    extras_require={
        'tf': ['tensorflow>=2.0.0'],
        'tf_gpu': ['tensorflow-gpu>=2.0.0'],
    },
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'jupyter',
    ],
)
