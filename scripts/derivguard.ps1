param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$Repository = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repository '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found: $Python"
}
& $Python -m derivguard @Arguments
exit $LASTEXITCODE
