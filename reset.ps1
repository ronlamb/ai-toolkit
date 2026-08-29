# 1. Clear Python's Bytecode Cache (__pycache__)
Write-Host "Cleaning Python __pycache__ directories..." -ForegroundColor Cyan
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue | 
    Remove-Item -Force -Recurse

# 2. Clear PyTorch's Triton Kernel Disk Cache
$TritonCache = Join-Path $env:USERPROFILE ".triton\cache"
if (Test-Path $TritonCache) {
    Write-Host "Cleaning Triton Kernel Cache..." -ForegroundColor Cyan
    Remove-Item -Path $TritonCache -Force -Recurse
}

# 3. Clear NVIDIA's Driver Compute Cache
# Windows stores CUDA/OpenGL shader caches in the Local AppData folder
$NvidiaCache = Join-Path $env:LOCALAPPDATA "NVIDIA\ComputeCache"
if (Test-Path $NvidiaCache) {
    Write-Host "Cleaning NVIDIA CUDA Compute Cache..." -ForegroundColor Cyan
    # Enumerate subfolders and remove them (bypasses locked file errors on the root folder)
    Get-ChildItem -Path $NvidiaCache -Directory | 
        Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
}

Write-Host "Cache cleanup complete! Restart your PyTorch script." -ForegroundColor Green