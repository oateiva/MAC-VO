<#
.SYNOPSIS
    Build and install a GTSAM Python wheel on Windows from a source tag.

.DESCRIPTION
    PyPI publishes no Windows wheels for GTSAM (Linux/macOS only), so on Windows
    the Python bindings must be built from source. This script:
      1. fetches the requested tag into an existing gtsam clone,
      2. checks it out in a separate git worktree (leaves your clone untouched),
      3. configures a Release build with MSVC + Ninja mirroring the official
         cibuildwheel flags (.github/scripts/python_wheels/cibw_before_all.sh),
         except boost-free (GTSAM_ENABLE_BOOST_SERIALIZATION=OFF) and shared
         libs, which the 4.3a2 python/CMakeLists.txt requires on Windows —
         it bundles gtsam.dll next to the .pyd via $<TARGET_RUNTIME_DLLS>,
      4. builds a cp3XX win_amd64 wheel,
      5. installs it into the target Python environment.

    Requires: VS 2022 (C++ toolset), CMake >= 3.22, Ninja, git.
    The target env must have the wrap deps: pip install -r python/dev_requirements.txt

.EXAMPLE
    powershell -File Scripts/build_gtsam_windows.ps1 `
        -Tag 4.3a2 `
        -GtsamClone C:\Users\oat\Documents\Github\gtsam `
        -PythonExe C:\Users\oat\.conda\envs\AITraining12\python.exe
#>
param(
    [Parameter(Mandatory = $true)] [string]$Tag,
    [Parameter(Mandatory = $true)] [string]$GtsamClone,
    [Parameter(Mandatory = $true)] [string]$PythonExe,
    [string]$VcVars = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    # Optional git patch applied to the worktree before building, e.g.
    # Scripts\patches\gtsam-posetopoint-wrapper.patch (wraps the C++
    # PoseToPointFactor for Python — required by factor_type: pose2point_native).
    [string]$PatchFile,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$worktree = Join-Path (Split-Path $GtsamClone -Parent) "gtsam-$Tag"
$build = Join-Path $worktree "build"
$pyVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

# 1+2. Fetch the tag and check it out in a worktree
git -C $GtsamClone fetch origin --tags
if (-not (Test-Path $worktree)) {
    git -C $GtsamClone worktree add $worktree $Tag
    if ($LASTEXITCODE -ne 0) { throw "git worktree add failed" }
}

if ($PatchFile) {
    $patch = (Resolve-Path $PatchFile).Path
    git -C $worktree apply --reverse --check $patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Patch already applied: $patch"
    } else {
        git -C $worktree apply $patch
        if ($LASTEXITCODE -ne 0) { throw "git apply failed for $patch" }
    }
}

# Wrap-tool deps must be importable by the target python during the build
& $PythonExe -m pip install -r (Join-Path $worktree "python\dev_requirements.txt")

# 3. Configure (MSVC via vcvars64, Ninja, Release)
# NOTE: GTSAM's wrap tooling unsets Python_EXECUTABLE from the CMake cache on
# every configure (wrap/cmake/GtwrapUtils.cmake) and re-runs find_package(Python
# <ver> EXACT), so pinning the executable directly does NOT stick — without the
# hints below it will happily link against some other Python on the machine
# (e.g. a Store Python) while still stamping the wheel with the target version.
# Python_ROOT_DIR + FIND_REGISTRY=NEVER + putting the env first on PATH survive
# that unset and force discovery of the intended interpreter.
$pyRoot = Split-Path $PythonExe -Parent
$configure = @(
    "set `"PATH=$pyRoot;$pyRoot\Scripts;%PATH%`" &&",
    "cmake $worktree -B $build -G Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION=$pyVersion",
    "-DPython_ROOT_DIR=$pyRoot",
    "-DPython_FIND_REGISTRY=NEVER -DPython_FIND_STRATEGY=LOCATION",
    "-DPYTHON_EXECUTABLE=$PythonExe",
    "-DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF",
    "-DGTSAM_BUILD_UNSTABLE=ON -DGTSAM_UNSTABLE_BUILD_PYTHON=ON",
    "-DGTSAM_USE_QUATERNIONS=OFF -DGTSAM_WITH_TBB=OFF",
    "-DGTSAM_ALLOW_DEPRECATED_SINCE_V43=OFF",
    "-DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF -DGTSAM_USE_BOOST_FEATURES=OFF",
    "-DCMAKE_INSTALL_PREFIX=$worktree\gtsam_install"
) -join " "
cmd /c "`"$VcVars`" >nul 2>&1 && $configure"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

# 4. Build (long: 30-60 min) and package the wheel
cmd /c "`"$VcVars`" >nul 2>&1 && cmake --build $build"
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Push-Location (Join-Path $build "python")
try {
    # Generate .pyi stubs into the package before packaging — official PyPI
    # wheels ship these (cibw runs the python-stubs target), and without them
    # pyright resolves gtsam to an opaque .pyd and flags every attribute access.
    # Args mirror python/CMakeLists.txt's python-stubs target.
    $stubArgs = @("-o", ".",
        "--enum-class-locations", "KernelFunctionType|NoiseFormat:gtsam.gtsam",
        "--enum-class-locations", "OrderingType:gtsam.gtsam.Ordering",
        "--numpy-array-use-type-var", "--ignore-all-errors")
    & $PythonExe -m pybind11_stubgen @stubArgs gtsam
    & $PythonExe -m pybind11_stubgen @stubArgs gtsam_unstable

    & $PythonExe setup.py bdist_wheel
    if ($LASTEXITCODE -ne 0) { throw "bdist_wheel failed" }
    $wheel = Get-ChildItem "dist\gtsam-*.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
    Write-Host "Wheel: $($wheel.FullName)"
} finally {
    Pop-Location
}

# Guard against the silent wrong-Python link described above: the .pyd must
# import against pythonXY.dll of the *target* interpreter.
& $PythonExe -c @"
import zipfile, sys, re
tag = f'python{sys.version_info.major}{sys.version_info.minor}'
data = zipfile.ZipFile(r'$($wheel.FullName)').read('gtsam/gtsam_py.pyd')
sys.exit(0 if tag.encode() in data else 1)
"@
if ($LASTEXITCODE -ne 0) { throw "Wheel's gtsam_py.pyd is not linked against the target Python — CMake found a different interpreter." }

# 5. Install into the target env
if (-not $SkipInstall) {
    # If gtsam came from conda (e.g. conda-forge 4.2.0), remove that package
    # first: conda remove -n <env> --force gtsam
    & $PythonExe -m pip install --force-reinstall --no-deps $wheel.FullName
    # The third-party gtsam-stubs package describes a pre-4.3 API and would
    # shadow the wheel's own bundled stubs — drop it if present.
    & $PythonExe -m pip uninstall -y gtsam-stubs 2>$null
    & $PythonExe -c "import gtsam, importlib.metadata as m; print('installed gtsam', m.version('gtsam'))"
}
