import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gans", # Replace with your own username
    version="0.0.1",
    author="Sanidhya Mangal",
    author_email="mangalsanidhya19@gmail.com",
    description="A light weight framework for tooling GANs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sanidhyamangal/gans",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow GAN generative model machine learning',
    python_requires='>=3.6',
)
