#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

CONDA_ROOT=/opt/conda
ENV_PREFIX=/opt/conda/envs/pdebench
SOURCE_DIR=/opt/src/sissopp
KOKKOS_SOURCE_DIR=/opt/src/kokkos
INSTALL_PREFIX=/opt/sissopp
SISSOPP_VERSION=v1.2.6
MINIFORGE_INSTALLER=/tmp/Miniforge3-Linux-x86_64.sh
SOURCE_ARCHIVE=/root/sissopp-v1.2.6.tar.gz
KOKKOS_SOURCE_ARCHIVE=/root/kokkos-486cc745.tar.gz

cleanup_downloads() {
    rm -f -- "$MINIFORGE_INSTALLER"
}
trap cleanup_downloads EXIT

echo '[1/7] Installing Ubuntu build dependencies'
if [[ "${SKIP_SYSTEM_DEPS:-0}" != 1 ]]; then
    apt-get -o Acquire::Retries=5 update
    apt-get -o Acquire::Retries=5 install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        clang-14 \
        cmake \
        coinor-libclp-dev \
        coinor-libcoinutils-dev \
        curl \
        gfortran \
        git \
        libboost-filesystem-dev \
        libboost-mpi-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libbz2-dev \
        libfmt-dev \
        libnlopt-cxx-dev \
        libomp-14-dev \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        pkg-config \
        zlib1g-dev
    rm -rf -- /var/lib/apt/lists/*
else
    echo 'System dependency installation already completed; skipping.'
fi

echo '[2/7] Installing Miniforge'
if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    curl --fail --location --retry 5 --retry-delay 3 --connect-timeout 15 --max-time 600 \
        https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh \
        --output "$MINIFORGE_INSTALLER"
    bash "$MINIFORGE_INSTALLER" -b -p "$CONDA_ROOT"
fi

echo '[3/7] Creating the pdebench Python 3.9.23 environment'
if [[ -x "$ENV_PREFIX/bin/python" ]] \
    && [[ "$($ENV_PREFIX/bin/python -c 'import platform; print(platform.python_version())')" == 3.9.23 ]]; then
    echo 'Existing pdebench environment already uses Python 3.9.23; reusing it.'
else
    if [[ -d "$ENV_PREFIX" ]]; then
        "$CONDA_ROOT/bin/conda" env remove --prefix "$ENV_PREFIX" --yes
    fi
    # The smaller Anaconda main index stays below the 1 GiB cgroup limit.
    CONDA_NO_PLUGINS=true "$CONDA_ROOT/bin/conda" create --yes --solver classic \
        --override-channels --channel https://repo.anaconda.com/pkgs/main \
        --name pdebench python=3.9.23 pip
fi

echo '[4/7] Installing pinned Python dependencies'
"$ENV_PREFIX/bin/python" -m pip install --no-cache-dir --upgrade \
    pip==25.2 setuptools==80.9.0 wheel==0.45.1
"$ENV_PREFIX/bin/python" -m pip install --no-cache-dir \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    pandas==2.2.3 \
    pybind11==2.13.6 \
    scikit-learn==1.6.1 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    toml==0.10.2

echo '[5/7] Configuring SISSO++ v1.2.6'
mkdir -p /opt/src
if [[ -e "$SOURCE_DIR" ]]; then
    case "$SOURCE_DIR" in
        /opt/src/sissopp) rm -rf -- "$SOURCE_DIR" ;;
        *) echo "Refusing unsafe source path: $SOURCE_DIR" >&2; exit 1 ;;
    esac
fi
if [[ -e "$INSTALL_PREFIX" ]]; then
    case "$INSTALL_PREFIX" in
        /opt/sissopp) rm -rf -- "$INSTALL_PREFIX" ;;
        *) echo "Refusing unsafe install path: $INSTALL_PREFIX" >&2; exit 1 ;;
    esac
fi
if [[ -f "$SOURCE_ARCHIVE" ]]; then
    mkdir -p "$SOURCE_DIR"
    tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE_DIR"
else
    git clone --depth 1 --branch "$SISSOPP_VERSION" --single-branch \
        https://gitlab.com/sissopp_developers/sissopp.git "$SOURCE_DIR"
fi

KOKKOS_CMAKE_ARGS=()
if [[ -f "$KOKKOS_SOURCE_ARCHIVE" ]]; then
    if [[ -e "$KOKKOS_SOURCE_DIR" ]]; then
        case "$KOKKOS_SOURCE_DIR" in
            /opt/src/kokkos) rm -rf -- "$KOKKOS_SOURCE_DIR" ;;
            *) echo "Refusing unsafe Kokkos path: $KOKKOS_SOURCE_DIR" >&2; exit 1 ;;
        esac
    fi
    mkdir -p "$KOKKOS_SOURCE_DIR"
    tar -xzf "$KOKKOS_SOURCE_ARCHIVE" --strip-components=1 -C "$KOKKOS_SOURCE_DIR"
    KOKKOS_CMAKE_ARGS+=("-DFETCHCONTENT_SOURCE_DIR_KOKKOS=$KOKKOS_SOURCE_DIR")
fi

PYBIND11_DIR="$($ENV_PREFIX/bin/python -m pybind11 --cmakedir)"
PYTHON_INSTDIR="$($ENV_PREFIX/bin/python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_FLAGS_RELEASE='-O0 -g0 -DNDEBUG' \
    -DCMAKE_CXX_FLAGS_RELEASE='-O0 -g0 -DNDEBUG -ftrack-macro-expansion=0 --param=ggc-min-expand=10 --param=ggc-min-heapsize=32768' \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DEXTERNAL_BUILD_N_PROCS=1 \
    -DPython_EXECUTABLE="$ENV_PREFIX/bin/python" \
    -DPYTHON_EXECUTABLE="$ENV_PREFIX/bin/python" \
    -DPYTHON_INSTDIR="$PYTHON_INSTDIR" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DSISSO_BUILD_EXE=ON \
    -DSISSO_BUILD_PARAMS=ON \
    -DSISSO_BUILD_PYTHON=ON \
    -DSISSO_BUILD_TESTS=OFF \
    "${KOKKOS_CMAKE_ARGS[@]}"

echo '[6/7] Building and installing SISSO++ with one compiler job'
# GCC can compile the core below 1 GiB with aggressive garbage collection.
cmake --build "$SOURCE_DIR/build" --target libsisso --parallel 1

# The generated pybind11 bindings exceed 1 GiB in GCC even at -O0. Compile only
# that target with Clang, without enabling a second OpenMP runtime. The final
# extension is still linked by GCC against libgomp and the common libstdc++ ABI.
PY_BIND_BUILD_MAKE="$SOURCE_DIR/build/src/CMakeFiles/_sissopp.dir/build.make"
PY_BIND_FLAGS="$SOURCE_DIR/build/src/CMakeFiles/_sissopp.dir/flags.make"
test -f "$PY_BIND_BUILD_MAKE"
test -f "$PY_BIND_FLAGS"
sed -i 's#/usr/bin/g++#/usr/bin/clang++-14#g' "$PY_BIND_BUILD_MAKE"
sed -i \
    -e 's/ -fopenmp / /g' \
    -e 's/ -ftrack-macro-expansion=0//g' \
    -e 's/ --param=ggc-min-expand=10//g' \
    -e 's/ --param=ggc-min-heapsize=32768//g' \
    "$PY_BIND_FLAGS"

cmake --build "$SOURCE_DIR/build" --parallel 1
cmake --install "$SOURCE_DIR/build"

echo '[7/7] Activating and validating the installed environment'
cat > /opt/activate-pdebench.sh <<'EOF'
source /opt/conda/etc/profile.d/conda.sh
conda activate pdebench
export PATH="/opt/sissopp/bin:$PATH"
export LD_LIBRARY_PATH="/opt/sissopp/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
EOF
chmod 0644 /opt/activate-pdebench.sh
if ! grep -Fq '. /opt/activate-pdebench.sh' /root/.bashrc; then
    printf '\n# Activate the SISSO++ benchmark environment.\n. /opt/activate-pdebench.sh\n' \
        >> /root/.bashrc
fi

source /opt/activate-pdebench.sh
test "$(python -c 'import platform; print(platform.python_version())')" = '3.9.23'
python -c "import sissopp, _sissopp; print('SISSO++ Python bindings imported successfully')"
test -x /opt/sissopp/bin/sisso++

{
    echo "python=$(python -c 'import platform; print(platform.python_version())')"
    echo "python_executable=$(command -v python)"
    echo "sissopp_tag=$SISSOPP_VERSION"
    echo "sissopp_executable=/opt/sissopp/bin/sisso++"
    echo "pybind11=$(python -c 'import pybind11; print(pybind11.__version__)')"
    echo "cmake=$(cmake --version | head -1)"
    echo "compiler=$(g++ --version | head -1)"
} > /opt/sissopp-build-info.txt

rm -rf -- "$SOURCE_DIR" "$KOKKOS_SOURCE_DIR"
rmdir /opt/src 2>/dev/null || true
"$CONDA_ROOT/bin/conda" clean --all --yes
rm -f -- /root/install_in_ubuntu22_pdebench.sh "$SOURCE_ARCHIVE" "$KOKKOS_SOURCE_ARCHIVE"

echo 'SISSO++ installation completed successfully.'
cat /opt/sissopp-build-info.txt
