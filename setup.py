from setuptools import setup, find_packages

setup(
    name='rob_pitch',
    version='0.1.2',
    author='Xinsheng Wang, Mingqi Jiang',
    author_email='w.xinshawn@gmail.com, mingqi.jiang@mobvoi.com',
    packages=find_packages(),
    description='Robust pitch prediction using PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'hydra-core==1.3.2',
        'modelscope==1.18.0',
        'numpy==1.24.3',
        'omegaconf==2.3.0',
        'setuptools==69.5.1',
        'soundfile==0.12.1',
        'soxr==0.3.7',
        'torch==2.0.1',
        'torchaudio==2.0.2',
        'tqdm==4.66.5'
    ],
    package_data={
        '': ['model.bin', 'config.yaml'],
    },
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Software Development :: Libraries",
    ],
)
