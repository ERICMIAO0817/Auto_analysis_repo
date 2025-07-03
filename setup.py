"""
GitAnalytics安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

# 读取requirements文件
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding='utf-8').strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="gitanalytics",
    version="1.0.0",
    author="GitAnalytics Team",
    author_email="contact@gitanalytics.com",
    description="A comprehensive Git repository analysis tool based on PyDriller",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gitanalytics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Version Control :: Git",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "full": [
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "gitanalytics=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="git analysis repository mining software engineering pydriller",
    project_urls={
        "Bug Reports": "https://github.com/your-username/gitanalytics/issues",
        "Source": "https://github.com/your-username/gitanalytics",
        "Documentation": "https://gitanalytics.readthedocs.io/",
    },
) 