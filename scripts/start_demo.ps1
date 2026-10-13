# MedQA-LangGraph Demo 一键启动（Windows）
# 用法：powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Test-Api {
  try {
    Invoke-WebRequest "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
    return $true
  } catch { return $false }
}
function Test-Web {
  try {
    Invoke-WebRequest "http://127.0.0.1:5173" -UseBasicParsing -TimeoutSec 2 | Out-Null
    return $true
  } catch { return $false }
}

# 1. 后端 API（:8000）
if (-not (Test-Api) ) {
  Write-Host "[1/3] Installing API dependencies..." -ForegroundColor Cyan
  pip install -q -r (Join-Path $root "api\requirements.txt")
  Write-Host "[2/3] Starting API server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
  Start-Process -NoNewWindow -FilePath python -ArgumentList "-m","uvicorn","api.main:app","--port","8000" -WorkingDirectory $root
  Start-Sleep -Seconds 4
  if (-not (Test-Api) ) { Write-Warning "API 未在 4 秒内就绪，可能仍在启动（首次需构建 22,571 条数据的内存索引，约 5-10 秒）" }
} else {
  Write-Host "[skip] API already running on :8000" -ForegroundColor DarkGray
}

# 2. 前端（:5173）
if (-not (Test-Web) ) {
  Write-Host "[3/3] Starting frontend on http://localhost:5173 ..." -ForegroundColor Cyan
  $fe = Join-Path $root "frontend"
  if (-not (Test-Path (Join-Path $fe "node_modules") )) {
    Write-Host "  first run: npm install ..." -ForegroundColor DarkGray
    Push-Location $fe; npm install | Out-Null; Pop-Location
  }
  Start-Process -NoNewWindow -FilePath cmd -ArgumentList "/c","npm run dev" -WorkingDirectory $fe
  Start-Sleep -Seconds 3
} else {
  Write-Host "[skip] frontend already running on :5173" -ForegroundColor DarkGray
}

Start-Process "http://localhost:5173"
Write-Host ""
Write-Host "Demo ready: http://localhost:5173" -ForegroundColor Green
Write-Host "  Tab1 数据集浏览  |  Tab2 多智能体流水线（mock 场景零密钥可演示）"
