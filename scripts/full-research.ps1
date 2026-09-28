$Repository = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repository '.venv\Scripts\python.exe'
& $Python -m derivguard full --profile research --device cuda --resume
exit $LASTEXITCODE
