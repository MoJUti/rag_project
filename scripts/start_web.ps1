param([int]$Port = 8765, [switch]$SkipBuild)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw '未找到项目虚拟环境，请先安装Python依赖。' }
if ($Port -lt 1 -or $Port -gt 65535) { throw '端口必须位于1至65535之间。' }
Push-Location $ProjectRoot
try {
    if (-not $SkipBuild) {
        if (-not (Test-Path -LiteralPath 'frontend/node_modules')) {
            & npm --prefix frontend ci
            if ($LASTEXITCODE -ne 0) { throw '前端依赖安装失败。' }
        }
        & npm --prefix frontend run build
        if ($LASTEXITCODE -ne 0) { throw '前端构建失败。' }
    }
    if (-not (Test-Path -LiteralPath 'frontend/dist/index.html')) { throw '前端产物不存在，不能跳过构建。' }
    Write-Host "新版知识库工作台：http://127.0.0.1:$Port/ （请保留此终端）"
    & $PythonPath -m uvicorn api.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'API启动失败，请检查依赖或端口占用。' }
} finally { Pop-Location }
