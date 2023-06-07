import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


__version__ = "0.0.1"

REPO_NAME = "text-summarizer-llm-project"
AUTHOR = "Debu Sinha"
AUTHOR_USER_NAME = "debu-sinha"
SRC_DIR = "text_summarizer"
AUTHOR_EMAIL = "debusinha2009@gmail.com"   


setuptools.setup(
    name=SRC_DIR,
    version=__version__,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description="NLP package for text summarization",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)