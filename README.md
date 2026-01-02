# Autonomous RL Trading Bot (v1)

## Quickstart (Windows PowerShell)

```powershell
cd "Autonomous RL Trading Bot"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pytest
python -c "import autonomous_rl_trading_bot; print(autonomous_rl_trading_bot.__version__)"



