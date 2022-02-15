import os

from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))


def read_version():
    data = {}
    path = os.path.join(ROOT, "synthtiger", "_version.py")
    with open(path, "r", encoding="utf-8") as fp:
        exec(fp.read(), data)
    return data["__version__"]


def read_long_description():
    path = os.path.join(ROOT, "README.md")
    with open(path, "r", encoding="utf-8") as fp:
        text = fp.read()
    return text


setup(
    name="synthtiger",
    version=read_version(),
    description="Synthetic text image generator for OCR model",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Moonbin Yim, Yoonsik Kim, Han-Cheol Cho, Sungrae Park",
    url="https://github.com/clovaai/synthtiger",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "arabic-reshaper",
        "blend-modes",
        "fonttools",
        "imgaug",
        "numpy",
        "opencv-python",
        "pillow>=8.2.0",
        "pygame",
        "python-bidi",
        "pytweening",
        "pyyaml",
        "regex",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "synthtiger = synthtiger.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
