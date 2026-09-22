#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

ENV_PREFIX=/opt/conda/envs/pdebench
SOURCE_ARCHIVE=/root/sissopp-v1.2.6-opt-source.tar.gz
SOURCE_DIR=/opt/src/sissopp-opt
INSTALL_PREFIX=/opt/sissopp
COMPILER_LAUNCHER=/tmp/sissopp-cxx-launcher.sh

test -x "$ENV_PREFIX/bin/python"
test -f "$SOURCE_ARCHIVE"
test -f "$INSTALL_PREFIX/lib/cmake/Kokkos/KokkosConfig.cmake"

apt-get -o Acquire::Retries=5 update
apt-get -o Acquire::Retries=5 install --yes --no-install-recommends \
    clang-14 \
    libomp-14-dev

if [[ -e "$SOURCE_DIR" ]]; then
    case "$SOURCE_DIR" in
        /opt/src/sissopp-opt) rm -rf -- "$SOURCE_DIR" ;;
        *) echo "Refusing unsafe source path: $SOURCE_DIR" >&2; exit 1 ;;
    esac
fi
mkdir -p "$SOURCE_DIR"
tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE_DIR"

cat > "$COMPILER_LAUNCHER" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

compiler=$1
shift
is_python_binding=0
is_feature_space=0
for arg in "$@"; do
    case "$arg" in
        */python/py_binding_cpp_def/*) is_python_binding=1 ;;
        */feature_creation/feature_space/FeatureSpace.cpp) is_feature_space=1 ;;
    esac
done

if [[ "$is_python_binding" == 1 ]]; then
    filtered_args=()
    for arg in "$@"; do
        case "$arg" in
            -fopenmp|-ftrack-macro-expansion=0|--param=ggc-min-expand=10|--param=ggc-min-heapsize=32768|-flto=*|-fno-fat-lto-objects) ;;
            *) filtered_args+=("$arg") ;;
        esac
    done
    exec /usr/bin/clang++-14 "${filtered_args[@]}" -O0 -g0
fi

if [[ "$is_feature_space" == 1 ]]; then
    exec "$compiler" "$@" -O0 -g0
fi

exec "$compiler" "$@"
EOF
chmod 0755 "$COMPILER_LAUNCHER"

PYBIND11_DIR="$($ENV_PREFIX/bin/python -m pybind11 --cmakedir)"
PYTHON_INSTDIR="$($ENV_PREFIX/bin/python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CXX_COMPILER_LAUNCHER="$COMPILER_LAUNCHER" \
    -DCMAKE_C_FLAGS_RELEASE='-O2 -DNDEBUG' \
    -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG -ftrack-macro-expansion=0 --param=ggc-min-expand=10 --param=ggc-min-heapsize=32768' \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DEXTERNAL_BUILD_N_PROCS=1 \
    -DKokkos_DIR="$INSTALL_PREFIX/lib/cmake/Kokkos" \
    -DPYTHON_EXECUTABLE="$ENV_PREFIX/bin/python" \
    -DPYTHON_INSTDIR="$PYTHON_INSTDIR" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DSISSO_BUILD_EXE=ON \
    -DSISSO_BUILD_PARAMS=ON \
    -DSISSO_BUILD_PYTHON=ON \
    -DSISSO_BUILD_TESTS=OFF

cmake --build "$SOURCE_DIR/build" --parallel 1
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    ctest --test-dir "$SOURCE_DIR/build" -R '^Regression$' --output-on-failure
cmake --install "$SOURCE_DIR/build"

echo 'Optimized SISSO++ rebuild completed successfully.'
