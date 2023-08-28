#! /bin/bash
# For now version 0.1.78
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install --upgrade --force-reinstall llama-cpp-python==0.1.78 --no-cache-dir