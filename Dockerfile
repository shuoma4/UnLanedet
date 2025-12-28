# Dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# 设置工作目录
WORKDIR /home

# 设置共享内存大小
ENV SHM_SIZE=20G

# 更换apt源为阿里云（加速下载）
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

# 安装系统工具（便于调试）
RUN apt-get update && apt-get install -y \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 设置pip镜像源（加速下载）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置默认命令
CMD ["/bin/bash"]