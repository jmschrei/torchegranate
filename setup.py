from setuptools import setup

setup(
	name='torchegranate',
	version='0.0.1',
	author='Jacob Schreiber',
	author_email='jmschreiber91@gmail.com',
	packages=['torchegranate'],
	url='https://github.com/jmschrei/torchegranate',
	license='LICENSE.txt',
	description='A rewrite of pomegranate using PyTorch.',
	install_requires=[
		'numpy >= 1.22.2', 
		'scipy >= 1.6.2',
		'scikit-learn >= 1.0.2',
		'torch >= 1.9.0'
	]
)