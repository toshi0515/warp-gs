<#
.SYNOPSIS
Patches, builds, and verifies the gsplat 1.5.3 CUDA extension on Windows.

.EXAMPLE
.\scripts\setup_gsplat_windows.ps1

.EXAMPLE
.\scripts\setup_gsplat_windows.ps1 -ForceRebuild -MaxJobs 4
#>

[CmdletBinding()]
param(
    [string]$Python,
    [ValidateRange(1, 64)]
    [int]$MaxJobs = 2,
    [switch]$ForceRebuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "This script only supports Windows."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python was not found at '$Python'. Run 'uv sync' first or pass -Python <path>."
}

$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$venvScripts = Split-Path -Parent $pythonPath
$env:PATH = "$venvScripts;$env:PATH"

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory)]
        [string]$Code,
        [string]$Description = "Python command"
    )

    $Code | & $pythonPath -
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

Write-Host "[gsplat] Inspecting $pythonPath"
$probeCode = @'
import json
from pathlib import Path

import gsplat
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access a CUDA device")

major, minor = torch.cuda.get_device_capability(0)
print(json.dumps({
    "arch": f"{major}.{minor}",
    "backend": str(Path(gsplat.__file__).resolve().parent / "cuda" / "_backend.py"),
    "device": torch.cuda.get_device_name(0),
    "gsplat": getattr(gsplat, "__version__", "unknown"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}))
'@
$probeLines = @($probeCode | & $pythonPath -)
if ($LASTEXITCODE -ne 0 -or $probeLines.Count -eq 0) {
    throw "Failed to inspect the Python, PyTorch, and gsplat environment."
}
$environmentInfo = $probeLines[-1] | ConvertFrom-Json
$backendPath = [string]$environmentInfo.backend
if (-not (Test-Path -LiteralPath $backendPath -PathType Leaf)) {
    throw "gsplat backend was not found at '$backendPath'."
}

Write-Host "[gsplat] gsplat $($environmentInfo.gsplat), torch $($environmentInfo.torch), CUDA $($environmentInfo.torch_cuda)"
Write-Host "[gsplat] GPU: $($environmentInfo.device) (SM $($environmentInfo.arch))"

# gsplat 1.5.3 passes GCC-only flags to MSVC. Keep this patch narrow and
# idempotent so a later gsplat version with the upstream fix is left alone.
$backendSource = [IO.File]::ReadAllText($backendPath)
$backendChanged = $false

if (-not $backendSource.Contains('cuda_opt_level = "-O0" if FAST_COMPILE else "-O3"')) {
    $flagsPattern = '(?m)^        opt_level = "-O0" if FAST_COMPILE else "-O3"\r?\n        extra_cflags = \[opt_level, "-Wno-attributes"\]\r?\n        extra_cuda_cflags = \[opt_level\]'
    $flagsReplacement = @'
        cuda_opt_level = "-O0" if FAST_COMPILE else "-O3"
        if os.name == "nt":
            # MSVC does not accept GCC flags such as -Wno-attributes, and its
            # optimization flags differ from nvcc's.
            extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]
        else:
            extra_cflags = [cuda_opt_level, "-Wno-attributes"]
        extra_cuda_cflags = [cuda_opt_level]
'@
    $patchedSource = [Text.RegularExpressions.Regex]::Replace(
        $backendSource,
        $flagsPattern,
        $flagsReplacement,
        [Text.RegularExpressions.RegexOptions]::None,
        [TimeSpan]::FromSeconds(1)
    )
    if ($patchedSource -eq $backendSource) {
        throw "Could not patch gsplat compiler flags. The installed gsplat layout is not recognized."
    }
    $backendSource = $patchedSource
    $backendChanged = $true
}

if (-not $backendSource.Contains('"GSPLAT_FORCE_REBUILD", "0"')) {
    $loaderPattern = '(?m)^        if os\.path\.exists\(os\.path\.join\(build_dir, f"\{name\}\.so"\)\) or os\.path\.exists\(\r?\n            os\.path\.join\(build_dir, f"\{name\}\.lib"\)\r?\n        \):'
    $loaderReplacement = @'
        if os.path.exists(os.path.join(build_dir, f"{name}.pyd")) and os.getenv(
            "GSPLAT_FORCE_REBUILD", "0"
        ) != "1":
            # A completed Windows build should not require cl.exe in every
            # Python process that loads the extension.
            _C = _import_module_from_library(name, build_dir, True)
        elif os.path.exists(os.path.join(build_dir, f"{name}.so")) or os.path.exists(
            os.path.join(build_dir, f"{name}.lib")
        ):
'@
    $patchedSource = [Text.RegularExpressions.Regex]::Replace(
        $backendSource,
        $loaderPattern,
        $loaderReplacement,
        [Text.RegularExpressions.RegexOptions]::None,
        [TimeSpan]::FromSeconds(1)
    )
    if ($patchedSource -eq $backendSource) {
        throw "Could not patch the gsplat Windows loader. The installed gsplat layout is not recognized."
    }
    $backendSource = $patchedSource
    $backendChanged = $true
}

if ($backendChanged) {
    [IO.File]::WriteAllText($backendPath, $backendSource, [Text.UTF8Encoding]::new($false))
    Write-Host "[gsplat] Patched $backendPath"
} else {
    Write-Host "[gsplat] Backend is already patched"
}

function Import-VisualStudioEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vcvarsCandidates = [Collections.Generic.List[string]]::new()
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $installations = @(
            & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        )
        $preferred = @($installations | Where-Object { $_ -match '\\2022\\' }) + @($installations)
        foreach ($installation in $preferred) {
            if ($installation) {
                $vcvarsCandidates.Add((Join-Path $installation "VC\Auxiliary\Build\vcvars64.bat"))
            }
        }
    }

    foreach ($edition in @("Community", "BuildTools", "Professional", "Enterprise")) {
        $vcvarsCandidates.Add("$env:ProgramFiles\Microsoft Visual Studio\2022\$edition\VC\Auxiliary\Build\vcvars64.bat")
        $vcvarsCandidates.Add("${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\$edition\VC\Auxiliary\Build\vcvars64.bat")
    }

    $vcvars = $vcvarsCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $vcvars) {
        throw "Visual Studio C++ Build Tools were not found. Install the MSVC v143 x64 tools first."
    }

    Write-Host "[gsplat] Loading MSVC environment from $vcvars"
    $developerEnvironment = @(& cmd.exe /d /c "`"$vcvars`" >nul && set")
    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64.bat failed with exit code $LASTEXITCODE."
    }
    foreach ($line in $developerEnvironment) {
        $name, $value = $line -split "=", 2
        if ($name -and $null -ne $value) {
            Set-Item -Path "Env:$name" -Value $value
        }
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "vcvars64.bat completed, but cl.exe is still unavailable."
    }
}

Import-VisualStudioEnvironment
$env:PATH = "$venvScripts;$env:PATH"

if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
    throw "ninja.exe is unavailable. Run 'uv sync' to install gsplat dependencies."
}
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
    throw "nvcc.exe is unavailable. Install the CUDA Toolkit matching PyTorch CUDA $($environmentInfo.torch_cuda)."
}

$env:MAX_JOBS = [string]$MaxJobs
$env:TORCH_CUDA_ARCH_LIST = [string]$environmentInfo.arch
Remove-Item Env:FAST_COMPILE -ErrorAction SilentlyContinue
Remove-Item Env:VERBOSE -ErrorAction SilentlyContinue
if ($ForceRebuild) {
    $env:GSPLAT_FORCE_REBUILD = "1"
    Write-Host "[gsplat] Rebuilding the optimized CUDA extension (MAX_JOBS=$MaxJobs)"
} else {
    Remove-Item Env:GSPLAT_FORCE_REBUILD -ErrorAction SilentlyContinue
    Write-Host "[gsplat] Loading or building the optimized CUDA extension (MAX_JOBS=$MaxJobs)"
}

$buildCode = @'
from gsplat.cuda._backend import _C

if _C is None:
    raise RuntimeError("gsplat CUDA extension did not load")
print(f"[gsplat] Extension: {getattr(_C, '__file__', None)}")
'@
Invoke-ProjectPython -Code $buildCode -Description "gsplat CUDA extension build"

Remove-Item Env:GSPLAT_FORCE_REBUILD -ErrorAction SilentlyContinue
Write-Host "[gsplat] Running CUDA forward/backward smoke test"
$verifyCode = @'
import torch
from gsplat.rendering import rasterization

device = "cuda"
means = torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.1, 2.5]], device=device, requires_grad=True)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
scales = torch.full((2, 3), 0.1, device=device, requires_grad=True)
opacities = torch.full((2,), 0.8, device=device, requires_grad=True)
colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, requires_grad=True)
viewmats = torch.eye(4, device=device)[None]
Ks = torch.tensor([[[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]]], device=device)

image, alpha, _ = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width=32, height=32
)
(image.sum() + alpha.sum()).backward()
torch.cuda.synchronize()

if not torch.isfinite(image).all() or means.grad is None or not torch.isfinite(means.grad).all():
    raise RuntimeError("gsplat produced non-finite output or gradients")
print(f"[gsplat] PASS image={tuple(image.shape)} alpha={tuple(alpha.shape)}")
'@
Invoke-ProjectPython -Code $verifyCode -Description "gsplat CUDA smoke test"

Write-Host "[gsplat] Setup completed successfully"
