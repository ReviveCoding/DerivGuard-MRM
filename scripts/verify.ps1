param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Continue"
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$artifactDir = Join-Path $repo "artifacts\tests"
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$pytestTemp = Join-Path ([IO.Path]::GetTempPath()) ("derivguard-pytest-" + $PID)
$junitPath = Join-Path $artifactDir "pytest_results.xml"

& $Python -m ruff format --check src tests scripts
$ruffFormat = $LASTEXITCODE
& $Python -m ruff check src tests scripts
$ruffCheck = $LASTEXITCODE
& $Python -m mypy src
$mypy = $LASTEXITCODE
& $Python -m pytest -q --basetemp $pytestTemp --junitxml $junitPath
$pytest = $LASTEXITCODE
$testCount = $null
$failureCount = $null
$errorCount = $null
$skippedCount = $null
if (Test-Path -LiteralPath $junitPath) {
    [xml]$junit = Get-Content -LiteralPath $junitPath -Raw
    $suite = $junit.testsuites.testsuite
    $testCount = [int]$suite.tests
    $failureCount = [int]$suite.failures
    $errorCount = [int]$suite.errors
    $skippedCount = [int]$suite.skipped
}
$env:DERIVGUARD_VERIFY_ROOT = $repo
$sourceTreeSha256 = & $Python -c "import os; from pathlib import Path; from derivguard.governance.gates import source_tree_sha256; print(source_tree_sha256(Path(os.environ['DERIVGUARD_VERIFY_ROOT'])))"
$sourceHashExit = $LASTEXITCODE

$summary = [ordered]@{
    schema_version = "1.1"
    captured_at = [DateTimeOffset]::UtcNow.ToString("o")
    python = (Resolve-Path -LiteralPath $Python).Path
    ruff_format_exit_code = $ruffFormat
    ruff_exit_code = $ruffCheck
    mypy_exit_code = $mypy
    pytest_exit_code = $pytest
    pytest_tests = $testCount
    pytest_failures = $failureCount
    pytest_errors = $errorCount
    pytest_skipped = $skippedCount
    pytest_junit = $junitPath
    pytest_basetemp = $pytestTemp
    source_tree_sha256 = if ($sourceHashExit -eq 0) { $sourceTreeSha256.Trim() } else { $null }
    status = if (($ruffFormat + $ruffCheck + $mypy + $pytest + $sourceHashExit) -eq 0) { "PASS" } else { "FAIL" }
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $artifactDir "final_test_summary.json") -Encoding utf8
if ($summary.status -ne "PASS") { exit 1 }
