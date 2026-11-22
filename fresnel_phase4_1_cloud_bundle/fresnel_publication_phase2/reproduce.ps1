param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$python = (Get-Command python).Source
$localPython = Join-Path $env:USERPROFILE 'AppData/Local/Python/pythoncore-3.14-64/python.exe'
if (Test-Path -LiteralPath $localPython) { $python=$localPython }
$base=Join-Path $env:USERPROFILE 'AppData/Local/Temp'
$libs=@('codex-fresnel-flint','codex-meep-validation-deps','codex-fresnel-sympy') | ForEach-Object {Join-Path $base $_} | Where-Object {Test-Path -LiteralPath $_}
$env:PYTHONPATH=(@($libs)+@($env:PYTHONPATH)) -join ';'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a new output folder.' }
$dest=New-Item -ItemType Directory -Path $OutputDirectory
Push-Location -LiteralPath $dest.FullName
try {
    foreach ($entry in @('run_pilot.py,runs','run_controls.py,controls','reference_benchmarks.py,reference_runs.json','legacy_control.py,legacy_run','stress_check.py,stress128')) {
        $parts=$entry.Split(',')
        & $python -B (Join-Path $PSScriptRoot $parts[0]) --out $parts[1]
        if ($LASTEXITCODE -ne 0) { throw "Failed $entry" }
    }
    & $python -B (Join-Path $PSScriptRoot 'summarize.py') --data . --out .
    if ($LASTEXITCODE -ne 0) { throw 'Result aggregation failed.' }
} finally { Pop-Location }

