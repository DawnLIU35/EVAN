from setuptools import find_packages, setup

setup(
    name="evan",
    version="0.1.0",
    description="Grounding Multimodal Reasoning with Evidence-Ablated Negatives",
    packages=find_packages(include=["evan", "evan.*", "verl", "verl.*"]),
    package_data={
        "evan": ["*.jinja", "config/*.yaml"],
        "verl": ["version/*", "trainer/*.yaml", "trainer/config/*.yaml", "trainer/config/*/*.yaml"],
    },
    python_requires=">=3.10",
)
