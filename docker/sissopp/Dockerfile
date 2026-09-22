# SISSO++ environment matching the Python version in kd-env.
#
# Build:
#   docker build -t sissopp-kd:1.2.6-py3.9.23 docker/sissopp
# Run interactively:
#   docker run --rm -it -v "${PWD}:/workspace" sissopp-kd:1.2.6-py3.9.23

FROM python:3.9.23-slim-bookworm

ARG SISSOPP_VERSION=v1.2.6
ARG BUILD_JOBS=2

LABEL org.opencontainers.image.title="SISSO++ for kd-env" \
      org.opencontainers.image.description="SISSO++ CLI and Python bindings on Python 3.9.23" \
      org.opencontainers.image.source="https://gitlab.com/sissopp_developers/sissopp" \
      org.opencontainers.image.version="${SISSOPP_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/sissopp/bin:${PATH}"

# SISSO++ requires CMake, a C++17 compiler, BLAS/LAPACK, MPI, and Boost.
# The remaining development libraries provide all optional CPU-side features,
# so CMake does not have to download and compile them itself.
RUN set -eux; \
    apt-get -o Acquire::Retries=5 update; \
    for attempt in 1 2 3 4 5; do \
      apt-get -o Acquire::Retries=5 install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        coinor-libclp-dev \
        coinor-libcoinutils-dev \
        gfortran \
        git \
        libboost-filesystem-dev \
        libboost-mpi-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libbz2-dev \
        libfmt-dev \
        libnlopt-cxx-dev \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        pkg-config \
        zlib1g-dev \
      && break; \
      if [ "${attempt}" -eq 5 ]; then exit 1; fi; \
      sleep 3; \
    done; \
    rm -rf /var/lib/apt/lists/*

# Versions are pinned to keep the Python 3.9 build reproducible. These satisfy
# the dependency versions declared by SISSO++ v1.2.6.
RUN python -m pip install --no-cache-dir --upgrade \
        pip==25.2 \
        setuptools==80.9.0 \
        wheel==0.45.1 \
    && python -m pip install --no-cache-dir \
        matplotlib==3.9.4 \
        numpy==2.0.2 \
        pandas==2.2.3 \
        pybind11==2.13.6 \
        scikit-learn==1.6.1 \
        scipy==1.13.1 \
        seaborn==0.13.2 \
        toml==0.10.2

RUN git clone --depth 1 --branch "${SISSOPP_VERSION}" --single-branch \
        https://gitlab.com/sissopp_developers/sissopp.git /opt/src/sissopp \
    && cmake -S /opt/src/sissopp -B /opt/src/sissopp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_C_FLAGS="-O3" \
        -DCMAKE_CXX_FLAGS="-O3" \
        -DCMAKE_INSTALL_PREFIX=/opt/sissopp \
        -DEXTERNAL_BUILD_N_PROCS="${BUILD_JOBS}" \
        -DPYTHON_INSTDIR="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')" \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
        -DSISSO_BUILD_EXE=ON \
        -DSISSO_BUILD_PARAMS=ON \
        -DSISSO_BUILD_PYTHON=ON \
        -DSISSO_BUILD_TESTS=OFF \
    && cmake --build /opt/src/sissopp/build --parallel "${BUILD_JOBS}" \
    && cmake --install /opt/src/sissopp/build \
    && test "$(python -c 'import platform; print(platform.python_version())')" = "3.9.23" \
    && python -c "import sissopp, _sissopp; print('SISSO++ Python bindings imported successfully')" \
    && command -v sisso++ \
    && rm -rf /opt/src/sissopp

RUN useradd --create-home --shell /bin/bash sissopp \
    && mkdir -p /workspace \
    && chown sissopp:sissopp /workspace

USER sissopp
WORKDIR /workspace

CMD ["/bin/bash"]
