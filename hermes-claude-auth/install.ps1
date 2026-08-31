#Requires -Version 5.1
<#
.SYNOPSIS
    Install hermes-claude-auth on Windows.
.DESCRIPTION
    Windows PowerShell equivalent of install.sh.
    Copies the billing bypass patch and installs the .pth loader hook into the hermes-agent venv.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -- Helpers ---------------------------------------------------------------
function Write-Ok   { param([string]$Msg) Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "[!!] $Msg" -ForegroundColor Yellow }
function Write-Err  { param([string]$Msg) Write-Host "[XX] $Msg" -ForegroundColor Red }

# -- Paths -----------------------------------------------------------------
$ScriptDir       = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Marker          = '# hermes-claude-auth managed'

# Locate the hermes root directory.  Search order:
#   1. $env:HERMES_HOME (explicit override)
#   2. %LOCALAPPDATA%\hermes   (Windows standard)
#   3. %USERPROFILE%\.hermes   (Linux/macOS convention)
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

$PatchesDir = if ($HermesDir) { Join-Path $HermesDir 'patches' } else { Join-Path $env:USERPROFILE '.hermes\patches' }

if (-not $HermesAgentDir) {
    Write-Err 'hermes-agent not found (checked %LOCALAPPDATA%\hermes\ and %USERPROFILE%\.hermes\)'
    Write-Host '    Install hermes-agent first: https://github.com/nousresearch/hermes-agent'
    Write-Host "    Or set HERMES_HOME to your hermes directory"
    exit 1
}

# -- Locate venv -----------------------------------------------------------
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

if (-not $VenvDir) {
    Write-Err "No virtualenv found in $HermesAgentDir (checked venv\, .venv\, and HERMES_VENV)"
    exit 1
}

# -- Locate venv Python (Windows layout: Scripts\python.exe) ---------------
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    # Fallback: maybe a Unix-style venv on WSL-created path
    $VenvPython = Join-Path $VenvDir 'bin\python.exe'
}
if (-not (Test-Path $VenvPython)) {
    $VenvPython = Join-Path $VenvDir 'bin\python3.exe'
}
if (-not (Test-Path $VenvPython)) {
    Write-Err "Python not found in venv at $VenvDir"
    exit 1
}

# -- Find site-packages ----------------------------------------------------
$SitePackages = & $VenvPython -c "import site; print(site.getsitepackages()[0] if site.getsitepackages() else site.getusersitepackages())" 2>$null
if (-not $SitePackages -or -not (Test-Path $SitePackages -PathType Container)) {
    Write-Err "site-packages directory does not exist: $SitePackages"
    exit 1
}

# -- Copy patch file -------------------------------------------------------
if (-not (Test-Path $PatchesDir -PathType Container)) {
    New-Item -ItemType Directory -Path $PatchesDir -Force | Out-Null
}
Copy-Item (Join-Path $ScriptDir 'anthropic_billing_bypass.py') (Join-Path $PatchesDir 'anthropic_billing_bypass.py') -Force
Write-Ok "Copied patch to $PatchesDir"

# -- Install hook via .pth shim --------------------------------------------
# Mirrors install.sh: a .pth file is processed by site.py on every platform,
# whereas a venv-local sitecustomize.py can be shadowed by a system-wide one.
$BootstrapName = '_hermes_claude_auth_bootstrap.py'
$PthName       = 'hermes_claude_auth.pth'
$BootstrapPath = Join-Path $SitePackages $BootstrapName
$PthPath       = Join-Path $SitePackages $PthName

Copy-Item (Join-Path $ScriptDir $BootstrapName) $BootstrapPath -Force
Write-Ok "Installed bootstrap module into $BootstrapPath"

Copy-Item (Join-Path $ScriptDir $PthName) $PthPath -Force
Write-Ok "Installed .pth shim into $PthPath"

# Migrate a legacy sitecustomize.py install: restore the user's original from
# backup when we have one, otherwise drop the now-superseded hook.  A
# sitecustomize.py we never wrote (no marker) is left untouched.
$Sitecustomize = Join-Path $SitePackages 'sitecustomize.py'
$LegacyBackup  = "$Sitecustomize.pre-hermes-claude-auth"
if ((Test-Path $Sitecustomize) -and ((Get-Content $Sitecustomize -Raw) -match [regex]::Escape($Marker))) {
    if (Test-Path $LegacyBackup) {
        Move-Item $LegacyBackup $Sitecustomize -Force
        Write-Warn "Migrated legacy sitecustomize.py install - restored your original from backup"
    }
    else {
        Remove-Item $Sitecustomize -Force
        Write-Warn "Migrated legacy sitecustomize.py install - removed superseded hook"
    }
}

# -- Windows credential mirror (Credential Manager → file) -----------------
# Analogous to the macOS Keychain mirror in install.sh.  Claude Code stores
# OAuth credentials as a Generic credential named 'Claude Code-credentials'
# via the keytar library.  hermes-agent reads ~/.claude/.credentials.json,
# so we mirror the Credential Manager entry into that file.
$CredDir  = Join-Path $env:USERPROFILE '.claude'
$CredFile = Join-Path $CredDir '.credentials.json'

$CredMirrored = $false
try {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class HermesCredManager {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct CREDENTIAL {
        public uint Flags;
        public uint Type;
        public string TargetName;
        public string Comment;
        public long LastWritten;
        public uint CredentialBlobSize;
        public IntPtr CredentialBlob;
        public uint Persist;
        public uint AttributeCount;
        public IntPtr Attributes;
        public string TargetAlias;
        public string UserName;
    }

    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool CredRead(string target, uint type, uint flags, out IntPtr credential);

    [DllImport("advapi32.dll")]
    private static extern void CredFree(IntPtr credential);

    public static string ReadGenericCredential(string target) {
        IntPtr credPtr;
        if (!CredRead(target, 1, 0, out credPtr)) return null;
        try {
            CREDENTIAL cred = (CREDENTIAL)Marshal.PtrToStructure(credPtr, typeof(CREDENTIAL));
            if (cred.CredentialBlobSize > 0 && cred.CredentialBlob != IntPtr.Zero) {
                byte[] blob = new byte[cred.CredentialBlobSize];
                Marshal.Copy(cred.CredentialBlob, blob, 0, (int)cred.CredentialBlobSize);
                return Encoding.UTF8.GetString(blob);
            }
            return null;
        } finally { CredFree(credPtr); }
    }
}
"@ -ErrorAction SilentlyContinue

    $CredSecret = [HermesCredManager]::ReadGenericCredential('Claude Code-credentials')
    if ($CredSecret) {
        if (-not (Test-Path $CredDir -PathType Container)) {
            New-Item -ItemType Directory -Path $CredDir -Force | Out-Null
        }
        $Existing = if (Test-Path $CredFile) { Get-Content $CredFile -Raw -ErrorAction SilentlyContinue } else { $null }
        if ($Existing -ne $CredSecret) {
            [System.IO.File]::WriteAllText($CredFile, $CredSecret)
            Write-Ok "Mirrored Claude Code credentials from Credential Manager to $CredFile"
        } else {
            Write-Ok 'Claude Code credentials file already matches Credential Manager'
        }
        $CredMirrored = $true
    }
} catch {
    # P/Invoke may fail in constrained language mode — fall through to warning
}

if (-not $CredMirrored -and -not (Test-Path $CredFile)) {
    Write-Warn "Credentials file not found at $CredFile"
    Write-Host '    Run: claude auth login --claudeai'
}

# -- No systemd on Windows; remind user to restart -------------------------
Write-Warn 'hermes-gateway must be restarted manually on Windows if currently running'

# -- Summary ---------------------------------------------------------------
Write-Host ''
Write-Ok 'Installation complete.'
Write-Host "  Patch:  $PatchesDir\anthropic_billing_bypass.py"
Write-Host "  Hook:   $Sitecustomize"
Write-Host "  Venv:   $VenvDir"
