from setuptools import find_packages, setup

setup(
    name="ean",
    version="0.1.0",
    description="Grounding Multimodal Reasoning with Evidence-Ablated Negatives",
    packages=find_packages(include=["ean", "ean.*", "verl", "verl.*"]),
    package_data={
        "ean": ["*.jinja", "config/*.yaml"],
        "verl": ["version/*", "trainer/*.yaml", "trainer/config/*.yaml", "trainer/config/*/*.yaml"],
    },
    python_requires=">=3.10",
)
