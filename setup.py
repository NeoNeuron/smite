import setuptools

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="smite",
    author="Kai Chen",
    author_email="kchen513@sjtu.edu.cn",

    version="0.0.2",
    url="https://github.com/NeoNeuron/smite",

    description="Package for symbolic transfer entropy and mutual information estimation.",

    install_requires=requirements,
    packages=setuptools.find_packages(),
)