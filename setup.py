from setuptools import setup, find_packages, Extension

setup(
    name="CDT",
    version="1.0.0",
    packages=find_packages(include=["CDTtools", "CDTtools.*"]),
    ext_modules=[
        Extension("dynlib", sources=["src/dynlib.c"]),
    ],
    entry_points={
        "console_scripts": [
            "CDT=GUI:main"
        ],
    },
    data_files=[
        ("CDT", [
            "saved_data/SST_arm_load",
            "saved_data/SST_axle_load",
            "saved_data/SST_proj_path",
            "saved_designs/ASAP.pkl",
            "saved_designs/SST.pkl",
            "saved_designs/SST_mk2.pkl",
            "saved_designs/simple_example.pkl",
        ]),
    ],
    install_requires=[
        "networkx==2.4",
        "numpy==1.18.5",
        "scipy==1.4.1",
        "matplotlib==3.2.1",
        "sympy==1.6",
    ],
    python_requires=">=3.8",
)
