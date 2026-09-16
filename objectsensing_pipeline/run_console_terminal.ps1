param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8770,
    [switch]$DebugLogs,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "ObjectSensing Run Console"
$python = $PythonPath
if ([string]::IsNullOrWhiteSpace($python)) {
    $python = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if (-not $command) {
            throw "Python was not found. Follow docs/SETUP.md or supply -PythonPath."
        }
        $python = $command.Source
    }
}

$arguments = @(
    (Join-Path $PSScriptRoot "run_console.py"),
    "--host", $ListenHost,
    "--port", $Port
)
if ($DebugLogs) {
    $arguments += "--debug"
}

Write-Host "Starting Run Console..." -ForegroundColor Cyan
& $python @arguments
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "Run Console stopped with exit code $exitCode." -ForegroundColor Red
}
