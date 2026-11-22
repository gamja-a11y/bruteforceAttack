param([string]$OutputDirectory = "replay_01")
$ErrorActionPreference = 'Stop'
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
$local = Join-Path $env:USERPROFILE 'AppData/Local/Python/pythoncore-3.14-64/python.exe'
if (Test-Path -LiteralPath $local) { $python = $local }
$depRoot = Join-Path $env:USERPROFILE 'AppData/Local/Temp'
$deps = @('codex-fresnel-flint','codex-meep-validation-deps','codex-fresnel-sympy') |
    ForEach-Object { Join-Path $depRoot $_ } | Where-Object { Test-Path -LiteralPath $_ }
$env:PYTHONPATH = (@($deps) + @($env:PYTHONPATH)) -join ';'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $python -B run_reaudit.py --out $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "Reaudit failed ($LASTEXITCODE)" }
} finally { Pop-Location }

