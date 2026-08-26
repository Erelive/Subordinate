# Launch the overlay, speak into it, and photograph the result.
#
# Captures only the bottom strip of the primary screen - enough to prove the
# caption box renders, positions and colours correctly, without photographing
# the whole desktop.
#
#   powershell -ExecutionPolicy Bypass -File tools\overlay_test.ps1

param(
    [int]$WarmupSec = 45,
    [int]$SpeakWaitSec = 7,
    [double]$StripFraction = 0.25,
    [string]$OutFile = "overlay_test.png",
    # Point these at dist\SubOrdinant\SubOrdinant.exe to test the built bundle
    # instead of the source tree.
    [string]$Exe = ".venv\Scripts\python.exe",
    # One string rather than an array: this script is usually invoked through
    # cmd.exe, which mangles array-valued arguments.
    [string]$LaunchArgs = "-u main.py --verbose"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "launching overlay: $Exe $LaunchArgs"
$startArgs = @{
    FilePath               = $Exe
    RedirectStandardOutput = "overlay_test.log"
    RedirectStandardError  = "overlay_test.err.log"
    PassThru               = $true
    WindowStyle            = "Hidden"
}
$argList = $LaunchArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
if ($argList.Count -gt 0) { $startArgs.ArgumentList = $argList }
$app = Start-Process @startArgs

try {
    Write-Host "waiting ${WarmupSec}s for model load and warmup..."
    Start-Sleep -Seconds $WarmupSec

    Write-Host "speaking..."
    $speak = Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "tools\speak.ps1", `
                      "-TextFile", "tools\sample_long.txt", "-Rate", "1" `
        -PassThru -WindowStyle Hidden

    Start-Sleep -Seconds $SpeakWaitSec

    Add-Type -AssemblyName System.Windows.Forms, System.Drawing

    # Without this the capture runs DPI-virtualised: Windows reports the scaled
    # desktop size and hands back a downscaled image, so the overlay lands in a
    # different place than the numbers suggest and looks misplaced when it is not.
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class NativeDpi {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
}
"@
    [NativeDpi]::SetProcessDPIAware() | Out-Null

    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $stripHeight = [int]($bounds.Height * $StripFraction)
    $top = $bounds.Height - $stripHeight

    $bitmap = New-Object System.Drawing.Bitmap($bounds.Width, $stripHeight)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen(0, $top, 0, 0, (New-Object System.Drawing.Size($bounds.Width, $stripHeight)))
    $bitmap.Save((Join-Path $root $OutFile), [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $bitmap.Dispose()
    Write-Host "saved $OutFile ($($bounds.Width)x$stripHeight)"

    if (-not $speak.HasExited) { Stop-Process -Id $speak.Id -Force -ErrorAction SilentlyContinue }
}
finally {
    if (-not $app.HasExited) { Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "overlay stopped"
}
