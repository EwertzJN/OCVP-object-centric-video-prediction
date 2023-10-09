from setuptools import setup, find_packages


setup(
        name="object-centric-video-prediction",
        version="0.1",
        packages=find_packages(),
        install_requires=[
            'fvcore',
            'imageio',
            'matplotlib',
            'numpy',
            'Pillow',
            'piqa',
            'scipy',
            'setuptools',
            'tensorflow',
            'tensorflow_datasets',
            'torch',
            'torchvision',
            'tqdm',
            'webcolors',
        ],
    )
