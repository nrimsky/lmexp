from setuptools import setup, find_packages

setup(
    name="lmexp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "anthropic==0.29.0",
        "bitsandbytes==0.42.0",
        "numpy==2.0.0",
        "python-dotenv==1.0.1",
        "requests==2.32.3",
        "torch==2.3.1",
        "transformers==4.41.2",
        "jupyter==1.0.0",
        "ipykernel==6.29.4",
        "accelerate==0.31.0",
        "pytest==8.2.2",
        "plotly==5.23.0",
    ],
)
