#Requires -Version 5.1
<#
.SYNOPSIS
    One-line remote installer for hermes-claude-auth on Windows.
.DESCRIPTION
    Usage: irm https://raw.githubusercontent.com/kristianvast/hermes-claude-auth/main/install-remote.ps1 | iex
#>

$ErrorActionPreference = 'Stop'

$Repo    = 'https://github.com/kristianvast/hermes-claude-auth.git'
$TmpDir  = Join-Path ([System.IO.Path]::GetTempPath()) "hermes-claude-auth-$(Get-Random)"

try {
    git clone --depth 1 $Repo $TmpDir 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[XX] git clone failed' -ForegroundColor Red
        exit 1
    }
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force -ErrorAction SilentlyContinue
    & (Join-Path $TmpDir 'install.ps1')
} finally {
    if (Test-Path $TmpDir) {
        Remove-Item $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
