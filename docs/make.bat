@ECHO OFF

set SOURCEDIR=.
set BUILDDIR=_build

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo The 'sphinx-build' command was not found. Install Sphinx first.
    exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%

:end
