@echo off
setlocal
cd /d %~dp0
echo Starting FireRedASR2S realtime server on http://localhost:8766
echo (models: FireRedASR2-AED + FireRedPunc + campplus voiceprint)
python server_firered.py
pause
