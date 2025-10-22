@echo off
set "self=%~nx0"
for %%F in (*) do if /i not "%%F"=="%self%" del /f /q "%%F"
for /d %%D in (*) do if /i not "%%D"=="%self%" rd /s /q "%%D"