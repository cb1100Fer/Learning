from setuptools import setup, find_packages

setup(
    name="tree-species-identifier",
    version="0.1.0",
    description="A framework for identifying tree species from images of leaves, bark, and seeds",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
