import subprocess
import os

subprocess.call(
    "pip install -U 'sapien>=3.0.0b1"
    , shell=True
)
subprocess.call(
    "pip install urdfpy usd-core open3d"
    , shell=True
)

subprocess.call(
    "pip uninstall networkx"
    , shell=True
)

subprocess.call(
    "pip install networkx"
    , shell=True
)

