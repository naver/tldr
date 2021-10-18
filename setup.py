from setuptools import setup


def readme():
    try:
        with open("README.md", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        # Python 2.7 doesn't support encoding argument in builtin open
        import io

        with io.open("README.md", encoding="UTF-8") as readme_file:
            return readme_file.read()


setup(
    name="TLDR",
    version="0.1.1",
    description="Twin Learning for Dimensionality Reduction",
    url="https://github.com/naver/tldr",
    long_description=readme(),
    long_description_content_type="text/x-rst",
    author="Jon Almazan",
    author_email="jon.almazan@naverlabs.com",
    license="CC BY-NCA-SA 4.0",
    packages=["tldr"],
    install_requires=[
        "rich",
        "numpy",
        "faiss>=1.7.0",
        "torch>=1.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Common Public License",
        "Environment :: GPU :: NVIDIA CUDA :: 11.2",
        "Programming Language :: Python :: 3.6",
    ],
)
