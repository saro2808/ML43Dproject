#!/bin/bash

pip install --upgrade pip

# trimesh 4.10.1 was used on local device with numpy 1.26.4 and python 3.11
# but the rest was done on colab with numpy 2.0.2 and python 3.12
# We use --extra-index-url to ensure we get the CUDA 12.6 versions of Torch/Pytorch3D
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --no-binary trimesh

# Handle the SymPy/Unsloth conflict
# We reinstall to ensure the 'printing' module is correctly mapped in Python 3.12
pip install --force-reinstall sympy==1.14.0

python3 -c "import trimesh; import numpy; print(f'SUCCESS: Trimesh {trimesh.__version__} compiled against NumPy {numpy.__version__}')"
