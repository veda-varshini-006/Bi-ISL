# Tasks runner for Bi-ISL on Windows PowerShell
param(
    [string]$Task = "verify"
)

switch ($Task) {
    "install" {
        pip install -e .[core,vision,training,evaluation,development,deployment,optionalresearch]
    }
    "verify" {
        python scripts/verify_environment.py
    }
    "lint" {
        ruff check src/ tests/
        mypy src/
    }
    "test" {
        pytest tests/
    }
    Default {
        Write-Host "Usage: .\tasks.ps1 [install|verify|lint|test]"
    }
}
