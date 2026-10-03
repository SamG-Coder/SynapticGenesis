param(
    [string]$Workspace = $PSScriptRoot,
    [string]$Runtime = (Join-Path $PSScriptRoot 'build/resident-conversation/synaptic-resident-conversation.exe'),
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Runtime)) { throw 'Build with ./build.ps1 -ResidentConversation -SkipTests, or supply -Runtime.' }
python (Join-Path $PSScriptRoot 'scripts/live_studio.py') --workspace $Workspace --runtime $Runtime --port $Port
