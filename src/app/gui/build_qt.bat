@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "%~dp0"
echo [1/3] qmake...
"C:\Qt\6.9.3\msvc2022_64\bin\qmake.exe" deeptracker_gui.pro
if errorlevel 1 exit /b 1
echo [2/3] nmake...
nmake
echo [3/3] done BUILD_EXIT=%errorlevel%
