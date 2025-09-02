### for CUDA 11.8, Python 3.10

#### Install Updates
```console
sudo apt update 
sudo apt install -y git wget curl zip unzip tmux htop build-essential ninja-build
```

#### Get Trellis
```console
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
```

#### Install Trellis
```console
. ./setup.sh --new-env --basic --xformers --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

#### Manual flash-attn wheel for pytorch 2.4  CUDA 11.8, Python 3.10
```console
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
```
