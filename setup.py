"""
pochitrain setup.py

pochitrainパッケージのインストール設定
"""

from setuptools import setup, find_packages
import os

# READMEファイルを読み込み


def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# requirements.txtを読み込み


def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()
                if line.strip() and not line.startswith('#')]


setup(
    name='pochitrain',
    version='0.1.0',
    author='Pochi Team',
    author_email='pochi@example.com',
    description='A tiny but clever CNN pipeline for images — as friendly as Pochi!',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/pochi-team/pochitrain',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='deep learning, computer vision, CNN, image classification, pytorch',
    python_requires='>=3.7',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'flake8>=3.8.0',
            'black>=21.0.0',
            'isort>=5.8.0',
            'pydocstyle>=6.0.0',
            'pre-commit>=2.12.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'pochi=pochitrain.cli.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
