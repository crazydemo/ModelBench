FROM ubuntu:latest
RUN apt-get update && \
    apt-get install -y \
    wget bzip2 build-essential \
    cmake numactl git vim

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/miniconda/bin:$PATH"
RUN conda init bash

RUN git clone --depth 1 --branch llvmorg-13.0.0  https://github.com/llvm/llvm-project /llvm-project
WORKDIR /llvm-project/build
RUN cmake ../llvm -DCMAKE_INSTALL_PREFIX=/llvm-project/install -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86
RUN make install -j

WORKDIR /home

CMD ["/bin/bash"]

