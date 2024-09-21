import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LISATracks",
    version="0.0.1",
    author="Diganta Bandopadhyay",
    author_email="diganta@star.sr.bham.ac.uk",
    description="Package to make animations of the strain of gravitational wave sources observable in LISA as a function of time.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dig07/LISATracks",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'matplotlib','scikit-learn', 'manim','jupyterlab'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
