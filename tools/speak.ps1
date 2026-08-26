# Speak text through the default output device using Windows SAPI.
# Gives the capture pipeline a repeatable audio source with known ground truth,
# so caption accuracy can be checked without a human talking.
#
#   powershell -ExecutionPolicy Bypass -File tools\speak.ps1 -Text "hello" -Rate 0

param(
    [string]$Text = "The quick brown fox jumps over the lazy dog. Live captions are being generated from Windows system audio using a local speech recognition model.",
    [string]$TextFile = "",
    [int]$Rate = 0,
    [int]$DelayMs = 0
)

if ($TextFile -ne "") { $Text = [IO.File]::ReadAllText($TextFile) }

Add-Type -AssemblyName System.Speech
if ($DelayMs -gt 0) { Start-Sleep -Milliseconds $DelayMs }
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = $Rate
$synth.Volume = 100
$synth.Speak($Text)
$synth.Dispose()
