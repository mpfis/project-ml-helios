import setuptools

setuptools.setup(
    name="cloudlogextractors",
    version="0.0.26",
    author="Max Fisher",
    description="Package contains wrapper methods around various APIs to extract log data from popular cloud log collection services like Azure Application Insights and AWS CloudWatch",
    packages=setuptools.find_packages(),
    python_requires=">=3.7"
)