$ErrorActionPreference = 'Stop'

# Run from the project root. Runtime requirements are required first; PyInstaller
# ships with requirements-dev.txt.
python -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { python -m pip install -r requirements-dev.txt }

# Bundle the trained evaluator so the exe auto-loads it and runs anywhere
# without the source tree or a model file next to it.
$addData = @()
if (Test-Path models\evaluator.npz) {
    $addData += '--add-data=models/evaluator.npz;models'
    Write-Host "Bundling models\evaluator.npz"
} else {
    Write-Host "No trained model found (models\evaluator.npz); building without EvalFile payload"
}

python -m PyInstaller --noconfirm --clean --onefile --name bc0 `
  --collect-all chess --collect-all numpy @addData engine.py

Write-Host "Built dist\bc0.exe"
