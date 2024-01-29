from setuptools import find_packages, setup

print(
    "Installing SEAR. Dependencies should already be installed with the provided conda env."
)

setup(
    name="sear",
    version="0.1.0",
    packages=find_packages(),
    description="Efficient RL via Disentangled Environment and Agent Representations",
    author="Kevin Gmelin, Shikhar Bahl, Russell Mendonca, and Deepak Pathak",
    author_email="kgmelin@andrew.cmu.edu",
)
