import sys
import os
from setuptools import setup, Extension
import pybind11

# --- CONFIGURATION ---
# If you get errors saying "Eigen/Dense" not found, add the path to Eigen here.
# Common paths: '/usr/include/eigen3', '/usr/local/include/eigen3'
# If you installed eigen via conda, it is often usually found automatically or in $CONDA_PREFIX/include/eigen3
eigen_include_path = '/usr/include/eigen3' 

# If you are using Conda, try to find Eigen automatically
if 'CONDA_PREFIX' in os.environ:
    possible_path = os.path.join(os.environ['CONDA_PREFIX'], 'include', 'eigen3')
    if os.path.exists(possible_path):
        eigen_include_path = possible_path

functions_module = Extension(
    name='tree',  # This must match the name in PYBIND11_MODULE(tree, m)
    sources=['tree.cpp'],
    include_dirs=[
        pybind11.get_include(),
        eigen_include_path
    ],
    language='c++',
    extra_compile_args=['-O3', '-std=c++11', '-fPIC'], # -O3 is crucial for MCMC speed!
)

setup(
    name='tree',
    version='1.0',
    description='C++ Tree for SMC',
    ext_modules=[functions_module],
)