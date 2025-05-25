# ===== setup.py =====
"""
Setup script for Active Inference + Diffusion package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="active-inference-diffusion",
    version="0.1.0",
    author="Zahra Sheikhbahaee",
    author_email="sheikhbahaeel@gmail.com",
    description="Active Inference with Diffusion Models for Continuous Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuronphysics/active-inference-diffusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "gymnasium>=0.28.0",
        "mujoco>=2.3.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "wandb>=0.13.0",
        "scipy>=1.7.0",
        "lz4>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
    },
    # Remove or fix the entry_points - there's no train:main function
    # entry_points={
    #     "console_scripts": [
    #         "aid-train=active_inference_diffusion.examples.train:main",
    #     ],
    # },
)