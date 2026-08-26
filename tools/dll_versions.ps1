# Report file versions of the MSVC runtime copies in play.
#
# Three copies can exist at once: the system one, the one PyQt6 vendors, and
# whatever PyInstaller collected into the bundle. When CTranslate2 binds to an
# older build than it was linked against, constructing a model faults.

$paths = @(
    "C:\Windows\System32\msvcp140.dll",
    "C:\Windows\System32\vcruntime140.dll",
    "$PSScriptRoot\..\.venv\Lib\site-packages\PyQt6\Qt6\bin\msvcp140.dll",
    "$PSScriptRoot\..\.venv\Lib\site-packages\PyQt6\Qt6\bin\vcruntime140.dll",
    "$PSScriptRoot\..\dist\SubOrdinant\_internal\msvcp140.dll",
    "$PSScriptRoot\..\dist\SubOrdinant\_internal\vcruntime140.dll"
)

foreach ($p in $paths) {
    if (Test-Path $p) {
        $item = Get-Item $p
        "{0,-14} {1,-16} {2}" -f $item.VersionInfo.FileVersion, ("{0:N0} bytes" -f $item.Length), $p
    }
    else {
        "{0,-14} {1,-16} {2}" -f "MISSING", "", $p
    }
}
