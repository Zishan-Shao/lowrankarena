# SVD-Benchmark Encoder Evaluation Docker Image
# Build from lowrankarena/ directory:
#   docker build -f eval_encoder/Dockerfile -t svd-benchmark:latest .
#
# Run interactive:
#   docker run --gpus all -it -v $(pwd):/workspace/lowrankarena svd-benchmark:latest
#
# Run GLUE benchmark:
#   docker run --gpus all -v $(pwd):/workspace/lowrankarena svd-benchmark:latest \
#     bash eval_encoder/scripts/one_click_glue.sh

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接,使 python 指向 python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# 设置工作目录
WORKDIR /workspace

# 复制 requirements 文件并安装依赖
COPY eval_encoder/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 复制整个代码库（按照 .dockerignore 过滤）
COPY . /workspace/lowrankarena/

# 设置工作目录到项目根目录
WORKDIR /workspace/lowrankarena

# 添加到 PYTHONPATH
ENV PYTHONPATH="/workspace/lowrankarena:${PYTHONPATH}"

# 创建输出目录
RUN mkdir -p \
    eval_encoder/models \
    eval_encoder/glue_results/logs \
    eval_encoder/eval_results

# 验证关键文件存在
RUN test -f eval_encoder/glue_pipeline.py && \
    test -f eval_encoder/run_encoder_benchmark.py && \
    test -f eval_encoder/load_compressed_model.py && \
    test -f eval_encoder/blocks.py && \
    test -f eval_encoder/flashsvd_backend.py && \
    test -f eval_encoder/scripts/one_click_glue.sh && \
    test -f eval_encoder/scripts/compare_all_methods.sh && \
    echo "✓ All core files verified"

# 设置脚本可执行权限
RUN chmod +x eval_encoder/scripts/*.sh

# 验证依赖
RUN python eval_encoder/check_dependencies.py

# 健康检查
HEALTHCHECK --interval=5m --timeout=3s \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# 设置默认入口点
ENTRYPOINT ["/bin/bash"]

# 使用说明
# ============================================================================
# 1. 构建镜像:
#    docker build -f eval_encoder/Dockerfile -t svd-benchmark:latest .
#
# 2. 交互式运行:
#    docker run --gpus all -it -v $(pwd):/workspace/lowrankarena svd-benchmark:latest
#
# 3. 运行 GLUE 评测 (Dense baseline):
#    docker run --gpus all -v $(pwd):/workspace/lowrankarena svd-benchmark:latest \
#      bash -c "cd /workspace/lowrankarena && METHOD=dense TASKS='sst2 mrpc' bash eval_encoder/scripts/one_click_glue.sh"
#
# 4. 运行 FWSVD 评测 (Full-matrix mode):
#    docker run --gpus all -v $(pwd):/workspace/lowrankarena svd-benchmark:latest \
#      bash -c "cd /workspace/lowrankarena && METHOD=fwsvd RANK_ATTN=256 RANK_FFN=256 RANK_WO=256 QKV_MODE=full TASKS='sst2' bash eval_encoder/scripts/one_click_glue.sh"
#
# 5. 运行对比测试:
#    docker run --gpus all -v $(pwd):/workspace/lowrankarena svd-benchmark:latest \
#      bash -c "cd /workspace/lowrankarena && bash eval_encoder/scripts/test_fwsvd_full_vs_perhead.sh"
# ============================================================================
