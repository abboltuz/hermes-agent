#Requires -Version 5.1
<#
.SYNOPSIS
    Uninstall hermes-claude-auth on Windows.
.DESCRIPTION
    Windows PowerShell equivalent of uninstall.sh.
    Removes the .pth loader hook; with -Purge also removes the patch file.
.PARAMETER Purge
    Also remove the patch file from ~/.hermes/patches/.
#>
param(
    [switch]$Purge
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Ok   { param([string]$Msg) Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "[!!] $Msg" -ForegroundColor Yellow }
function Write-Skip { param([string]$Msg) Write-Host "[--] $Msg" -ForegroundColor Yellow }

# -- Locate venv -----------------------------------------------------------
# Search order: $env:HERMES_HOME → %LOCALAPPDATA%\hermes → %USERPROFILE%\.hermes
$HermesDir = $null
$HermesAgentDir = $null

$CandidateDirs = @()
if ($env:HERMES_HOME -and (Test-Path $env:HERMES_HOME -PathType Container)) {
    $CandidateDirs += $env:HERMES_HOME
}
$CandidateDirs += Join-Path $env:LOCALAPPDATA 'hermes'
$CandidateDirs += Join-Path $env:USERPROFILE '.hermes'

foreach ($dir in $CandidateDirs) {
    $candidate = Join-Path $dir 'hermes-agent'
    if (Test-Path $candidate -PathType Container) {
        $HermesDir = $dir
        $HermesAgentDir = $candidate
        break
    }
}

$VenvDir = $null

if ($env:HERMES_VENV -and (Test-Path $env:HERMES_VENV -PathType Container)) {
    $VenvDir = $env:HERMES_VENV
}
elseif (Test-Path (Join-Path $HermesAgentDir 'venv') -PathType Container) {
    $VenvDir = Join-Path $HermesAgentDir 'venv'
}
elseif (Test-Path (Join-Path $HermesAgentDir '.venv') -PathType Container) {
    $VenvDir = Join-Path $HermesAgentDir '.venv'
}

$RemovedHook   = $false
$RestoredHook  = $false
$RemovedPatch  = $false

# -- Remove hook -----------------------------------------------------------
if (-not $VenvDir) {
    Write-Skip 'No hermes venv found, skipping hook removal'
}
else {
    $VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
    if (-not (Test-Path $VenvPython)) {
        $VenvPython = Join-Path $VenvDir 'bin\python.exe'
    }

    $SitePackages = $null
    if (Test-Path $VenvPython) {
        $SitePackages = & $VenvPython -c "import site; print(site.getsitepackages()[0])" 2>$null
    }

    if (-not $SitePackages) {
        Write-Skip 'Could not detect site-packages, skipping hook removal'
    }
    else {
        $PthPath       = Join-Path $SitePackages 'hermes_claude_auth.pth'
        $BootstrapPath = Join-Path $SitePackages '_hermes_claude_auth_bootstrap.py'
        $Sitecustomize = Join-Path $SitePackages 'sitecustomize.py'
        $BackupFile    = Join-Path $SitePackages 'sitecustomize.py.pre-hermes-claude-auth'

        if (Test-Path $PthPath) {
            Remove-Item $PthPath -Force
            Write-Ok "Removed .pth shim from $PthPath"
            $RemovedHook = $true
        }
        if (Test-Path $BootstrapPath) {
            Remove-Item $BootstrapPath -Force
            Write-Ok "Removed bootstrap module from $BootstrapPath"
            $RemovedHook = $true
        }

        if (Test-Path $Sitecustomize) {
            if ((Get-Content $Sitecustomize -Raw) -match [regex]::Escape('# hermes-claude-auth managed')) {
                if (Test-Path $BackupFile) {
                    Move-Item $BackupFile $Sitecustomize -Force
                    Write-Ok 'Restored original sitecustomize.py from legacy backup'
                    $RestoredHook = $true
                }
                else {
                    Remove-Item $Sitecustomize -Force
                    Write-Ok "Removed legacy sitecustomize.py hook from $SitePackages"
                    $RemovedHook = $true
                }
            }
            else {
                Write-Skip 'sitecustomize.py not ours'
            }
        }

        if (-not $RemovedHook -and -not $RestoredHook) {
            Write-Skip 'Hook not found (already removed)'
        }
    }
}

# -- Purge patch file ------------------------------------------------------
if ($Purge) {
    $PatchDir  = if ($HermesDir) { Join-Path $HermesDir 'patches' } else { Join-Path $env:USERPROFILE '.hermes\patches' }
    $PatchFile = Join-Path $PatchDir 'anthropic_billing_bypass.py'

    if (Test-Path $PatchFile) {
        Remove-Item $PatchFile -Force
        Write-Ok 'Removed patch from ~/.hermes/patches/'
        $RemovedPatch = $true
    }

    # Remove patches dir if empty
    if ((Test-Path $PatchDir -PathType Container) -and
        @(Get-ChildItem $PatchDir -Force).Count -eq 0) {
        Remove-Item $PatchDir -Force
    }
}

# -- Summary ---------------------------------------------------------------
Write-Host ''
Write-Ok 'Summary:'
if ($RestoredHook) {
    Write-Host '  - Restored sitecustomize.py from legacy backup'
}
elseif ($RemovedHook) {
    Write-Host '  - Removed .pth shim and bootstrap module'
}
else {
    Write-Host '  - No hook changes needed'
}
if ($RemovedPatch) {
    Write-Host '  - Removed patch file'
}
