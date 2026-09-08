$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
foreach ($testScript in @('audit_tests.py','tests\test_measurement.py','tests\test_scan_policy.py','tests\test_fund_engine.py','tests\test_archive_provenance.py','tests\test_fund_web.py','test_bulk_fundamentals.py','test_price_snapshot.py','test_trading_research.py','smoke_app.py')) {
    & '.\.venv\Scripts\python.exe' $testScript
    if ($LASTEXITCODE -ne 0) { throw "Failed: $testScript" }
}
Write-Host 'All offline checks passed.'
