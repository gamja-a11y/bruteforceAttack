param([string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
$localPython = Join-Path $env:USERPROFILE 'AppData/Local/Python/pythoncore-3.14-64/python.exe'
if (Test-Path -LiteralPath $localPython) { $python = $localPython }
$depRoot = Join-Path $env:USERPROFILE 'AppData/Local/Temp'
$libs = @('codex-fresnel-flint','codex-meep-validation-deps','codex-fresnel-sympy') | ForEach-Object { Join-Path $depRoot $_ } | Where-Object { Test-Path -LiteralPath $_ }
$env:PYTHONPATH = (@($libs) + @($env:PYTHONPATH)) -join ';'
Push-Location -LiteralPath $PSScriptRoot
try {
    $params = @('-B', 'independent_counterexample_tests.py')
    if ($OutputDirectory) { $params += @('--out', $OutputDirectory) }
    & $python @params
    if ($LASTEXITCODE -ne 0) { throw "Audit did not finish (exit $LASTEXITCODE)." }
} finally { Pop-Location }

