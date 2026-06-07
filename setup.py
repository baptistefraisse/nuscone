from setuptools import setup, find_packages

setup(
    name="nuscone",
    version="0.1.0",
    description="Neutron analysis for SCONE",
    packages=find_packages(include=["nuscone", "nuscone.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "black",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "nuscone=nuscone.cli:main",
        ],
    },
    python_requires=">=3.10",
)