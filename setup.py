from setuptools import setup, find_packages

setup(
  name = 'soft-moe-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.7',
  license='MIT',
  description = 'Soft MoE - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/soft-moe-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'mixture of experts'
  ],
  install_requires=[
    'einops>=0.6.1',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
