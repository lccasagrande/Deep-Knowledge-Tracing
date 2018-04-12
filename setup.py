import sys
from setuptools import find_packages, setup

CURRENT_PYTHON = sys.version_info.major
REQUIRED_PYTHON = 3

# This check and everything above must remain compatible with Python 2.7.
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
        ==========================
        Unsupported Python version
        ==========================
        
        This project requires Python {}.{}, but you're trying to install it on Python {}.{}.        
        """.format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))    

    sys.exit(1)

setup(name='DKT'
    , version=0.2
    , python_requires='>={}'.format(REQUIRED_PYTHON)
    , author='lccasagrande'
    , description=('An implementation of Deep Knowledge Tracing (DKT) with Keras and Tensorflow')
    , packages=find_packages()
    , include_package_data=True    
    , extras_require={
        'tf': ['tensorflow>=1.5.0'],
        'tf_gpu': ['tensorflow-gpu>=1.5.0'],
    }
    , install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'jupyter',
        'h5py>=2.7.1',
        'scikit-learn>=0.19.1',
        'keras>=2.1.3'
    ]
)
