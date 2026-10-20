#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

CONDA_ROOT=/opt/conda
ENV_PREFIX=/opt/conda/envs/pdebench
SOURCE_DIR=/opt/src/sissopp
KOKKOS_SOURCE_DIR=/opt/src/kokkos
INSTALL_PREFIX=/opt/sissopp
SISSOPP_VERSION=v1.2.6
SISSOPP_ARCHIVE=/root/sissopp-v1.2.6-source.tar.gz
KOKKOS_ARCHIVE=/root/kokkos-4.3.0-source.tar.gz
MINIFORGE_INSTALLER=/tmp/Miniforge3-Linux-x86_64.sh
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if (( BUILD_JOBS > 8 )); then
    BUILD_JOBS=8
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
export MAKEFLAGS="-j$BUILD_JOBS"

cleanup_downloads() {
    rm -f -- "$MINIFORGE_INSTALLER"
}
trap cleanup_downloads EXIT

echo '[1/8] Installing Ubuntu build dependencies'
apt-get -o Acquire::Retries=5 update
apt-get -o Acquire::Retries=5 install --yes --no-install-recommends \
    build-essential \
    ca-certificates \
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
    libopenblas-dev \
    libopenmpi-dev \
    openmpi-bin \
    pkg-config \
    zlib1g-dev

echo '[2/8] Installing Miniforge'
if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    curl --fail --location --retry 5 --retry-delay 3 --connect-timeout 15 --max-time 600 \
        https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh \
        --output "$MINIFORGE_INSTALLER"
    bash "$MINIFORGE_INSTALLER" -b -p "$CONDA_ROOT"
fi

echo '[3/8] Creating pdebench with Python 3.9.23'
if [[ -x "$ENV_PREFIX/bin/python" ]] \
    && [[ "$($ENV_PREFIX/bin/python -c 'import platform; print(platform.python_version())')" == 3.9.23 ]]; then
    echo 'Reusing the existing Python 3.9.23 environment.'
else
    if [[ -d "$ENV_PREFIX" ]]; then
        "$CONDA_ROOT/bin/conda" env remove --prefix "$ENV_PREFIX" --yes
    fi
    CONDA_NO_PLUGINS=true "$CONDA_ROOT/bin/conda" create --yes --solver classic \
        --override-channels --channel https://repo.anaconda.com/pkgs/main \
        --name pdebench python=3.9.23 pip
fi

echo '[4/8] Installing pinned Python dependencies'
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

echo '[5/8] Preparing pinned SISSO++ and Kokkos sources'
test -f "$SISSOPP_ARCHIVE"
test -f "$KOKKOS_ARCHIVE"
mkdir -p /opt/src
for target in "$SOURCE_DIR" "$KOKKOS_SOURCE_DIR" "$INSTALL_PREFIX"; do
    if [[ -e "$target" ]]; then
        case "$target" in
            /opt/src/sissopp|/opt/src/kokkos|/opt/sissopp) rm -rf -- "$target" ;;
            *) echo "Refusing unsafe path: $target" >&2; exit 1 ;;
        esac
    fi
done
mkdir -p "$SOURCE_DIR" "$KOKKOS_SOURCE_DIR"
tar -xzf "$SISSOPP_ARCHIVE" -C "$SOURCE_DIR"
tar -xzf "$KOKKOS_ARCHIVE" --strip-components=1 -C "$KOKKOS_SOURCE_DIR"

echo "[6/8] Building SISSO++ Release with $BUILD_JOBS jobs"
PYBIND11_DIR="$($ENV_PREFIX/bin/python -m pybind11 --cmakedir)"
PYTHON_INSTDIR="$($ENV_PREFIX/bin/python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_FLAGS_RELEASE='-O3 -DNDEBUG' \
    -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG' \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DEXTERNAL_BUILD_N_PROCS="$BUILD_JOBS" \
    -DFETCHCONTENT_SOURCE_DIR_KOKKOS="$KOKKOS_SOURCE_DIR" \
    -DPYTHON_EXECUTABLE="$ENV_PREFIX/bin/python" \
    -DPYTHON_INSTDIR="$PYTHON_INSTDIR" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DSISSO_BUILD_EXE=ON \
    -DSISSO_BUILD_PARAMS=ON \
    -DSISSO_BUILD_PYTHON=ON \
    -DSISSO_BUILD_TESTS=OFF
cmake --build "$SOURCE_DIR/build" --parallel "$BUILD_JOBS"

echo '[7/8] Running the upstream CLI regression test and installing'
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    ctest --test-dir "$SOURCE_DIR/build" -R '^Regression$' --output-on-failure
cmake --install "$SOURCE_DIR/build"

echo '[8/8] Activating and validating the installed environment'
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
python - <<'PY'
import numpy as np
import pybind11
import sissopp
import _sissopp
from sissopp.sklearn import SISSORegressor

x = np.linspace(-2.0, 2.0, 24)
X = np.column_stack((x, np.cos(x)))
y = 1.25 + 2.0 * x
model = SISSORegressor(
    prop_label="y",
    prop_unit="",
    n_dim=1,
    max_rung=0,
    n_sis_select=2,
    workdir="/tmp/sissopp-python-smoke",
    clean_workdir=True,
    verbose=False,
)
model.fit(X, y)
rmse = float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
assert rmse < 1e-12, rmse
print(f"SISSO++ Python fit passed; RMSE={rmse:.3e}")
PY
test -x "$INSTALL_PREFIX/bin/sisso++"
ldd "$INSTALL_PREFIX/bin/sisso++" | grep -q 'not found' && exit 1 || true
ldd "$PYTHON_INSTDIR/_sissopp.cpython-39-x86_64-linux-gnu.so" | grep -q 'not found' && exit 1 || true

{
    echo "os=$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"')"
    echo "python=$(python -c 'import platform; print(platform.python_version())')"
    echo "python_executable=$(command -v python)"
    echo "sissopp_tag=$SISSOPP_VERSION"
    echo 'sissopp_commit=fafd8a11e04241c5f1361dee5876ae3f84defcab'
    echo 'kokkos_version=4.3.0'
    echo 'kokkos_commit=486cc745cb9a287f3915061455105a3ee588c616'
    echo "sissopp_executable=$INSTALL_PREFIX/bin/sisso++"
    echo "pybind11=$(python -c 'import pybind11; print(pybind11.__version__)')"
    echo 'build_type=Release'
    echo 'optimization=-O3'
    echo "build_jobs=$BUILD_JOBS"
    echo 'runtime_threads_default=1'
    echo 'cli_regression_test=passed'
    echo 'python_fit_smoke_test=passed'
} > /opt/sissopp-build-info.txt

rm -rf -- "$SOURCE_DIR" "$KOKKOS_SOURCE_DIR" /tmp/sissopp-python-smoke
rmdir /opt/src 2>/dev/null || true
rm -f -- "$SISSOPP_ARCHIVE" "$KOKKOS_ARCHIVE" /root/install_release_ubuntu22_pdebench.sh
"$CONDA_ROOT/bin/conda" clean --all --yes
rm -rf -- /var/lib/apt/lists/*

echo 'SISSO++ Release installation completed successfully.'
cat /opt/sissopp-build-info.txt
