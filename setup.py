import setuptools

setuptools.setup(
    name="brain_utils",
    version="0.0.2",
    author="Kevin Faust",
    author_email="kevin.faust@mail.utoronto.ca",
    url="https://github.com/kevinfaust0308/brAIn-utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "fuzzywuzzy", "numpy", "matplotlib", "opencv-python", "Pillow==6.2.2", "algorithmia", "seaborn", "sendgrid"
    ],
    extras_require={
        "tf": ["tensorflow>=2.0,<2.1"],
        "tf_gpu": ["tensorflow-gpu>=2.0,<2.1"],
    }
)
