from setuptools import setup, find_packages

setup(
    name="rxvision-frontend",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.31.0",
        "Pillow==10.2.0",
        "requests==2.31.0",
        "reportlab==4.0.9",
        "numpy==1.26.3",
        "pandas==2.2.0"
    ],
    python_requires=">=3.9",
) 