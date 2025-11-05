# Start from Ubuntu 24.04 base image
FROM ubuntu:24.04

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites for micromamba and general tools
RUN apt-get update && apt-get install -y \
    curl bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set micromamba environment variables
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Create environment and install HMMER
RUN micromamba create -y -n bioinfo -c conda-forge -c bioconda hmmer && \
    micromamba clean --all --yes

# Activate the environment by default
SHELL ["micromamba", "run", "-n", "bioinfo", "/bin/bash", "-c"]

# Default command
CMD ["hmmscan", "-h"]