@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d C:\projects\aidaemon
cargo build 2>&1
echo EXIT_CODE=%ERRORLEVEL%
