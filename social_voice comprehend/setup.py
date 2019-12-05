from __future__ import division

from setuptools import find_packages, setup

from pathlib import Path

from setuptools import find_packages, setup

import os 


path = os.path.dirname(__file__)

requirements = os.path.join(path,'requirements.txt')

with open(requirements, mode='rt', encoding='utf-8') as fp:
    install_requires=[line.strip() for line in fp]

setup(
    name = 'NLP',
    version = '1.0.1',
    description = 'comments keywords extractions ' + ' comments sentiment analysis',
    author = 'LIN Xinrun',
    author_email = 'linxr6@outlook.com',
    keywords = ['NLP', 'keywords extractions', 'sentiment analysis'],
    packages = find_packages(exclude=('text*',)),
    package_data = {'':['*.txt', '*.pkl', '*.ttc', '*.xlsx']},
    include_package_data=True,
    install_requires=install_requires,

)


