$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskInterpreter = [System.IO.Path]::GetFullPath((Join-Path $taskRoot '../../../.venv/Scripts/python.exe'))
$taskTemp = Join-Path $taskRoot 'compiler_tmp'
New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$priorTemp = $env:TEMP
$priorTmp = $env:TMP
Push-Location $taskRoot
try {
    $env:TEMP = $taskTemp
    $env:TMP = $taskTemp
    & $taskInterpreter setup.py build_ext --inplace
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
    & $taskInterpreter validate.py
    if ($LASTEXITCODE -ne 0) { throw 'Validation failed' }
} finally {
    Pop-Location
    $env:TEMP = $priorTemp
    $env:TMP = $priorTmp
}
