@echo off
chcp 65001 >nul
cd /d "%~dp0relay"
where uv >nul 2>nul || (
  echo uv가 설치되어 있지 않습니다: https://docs.astral.sh/uv/
  pause
  exit /b 1
)
uv run arkit-fbt-relay %*
if errorlevel 1 pause
