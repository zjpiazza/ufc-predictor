from setuptools import setup, find_packages

setup(
    name="ufc_predictor",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'keras',
        'aiohttp',
        'beautifulsoup4',
        'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'ufc=ufc_predictor.cli.main:cli',
        ],
    },
) 