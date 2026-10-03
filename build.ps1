param([string]$Architecture='120', [switch]$SkipTests, [switch]$TraceDiagnostic, [switch]$AdaptationProbe, [switch]$ReplayPriorityProbe, [switch]$ReplaySelectionProbe, [switch]$CapacityProbe, [switch]$EarlyLearningProbe)
$ErrorActionPreference='Stop'
$diagnostics=@(
  @{Selected=$TraceDiagnostic; Name='trace-diagnostic'; Option='SG_BUILD_TRACE_DIAGNOSTIC'},
  @{Selected=$AdaptationProbe; Name='adaptation-probe'; Option='SG_BUILD_ADAPTATION_PROBE'},
  @{Selected=$ReplayPriorityProbe; Name='replay-priority-probe'; Option='SG_BUILD_REPLAY_PRIORITY_PROBE'},
  @{Selected=$ReplaySelectionProbe; Name='replay-selection-probe'; Option='SG_BUILD_REPLAY_SELECTION_PROBE'},
  @{Selected=$CapacityProbe; Name='capacity-probe'; Option='SG_BUILD_CAPACITY_PROBE'},
  @{Selected=$EarlyLearningProbe; Name='early-learning-probe'; Option='SG_BUILD_EARLY_LEARNING_PROBE'}
)
$enabledDiagnostics=@($diagnostics | Where-Object {$_.Selected})
if($enabledDiagnostics.Count -gt 1){throw 'Select one diagnostic build per invocation.'}
$root=$PSScriptRoot
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio C++ tools are required.' }
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$ninja=Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
if($enabledDiagnostics.Count -eq 1){
  # A separate output tree preserves the executable used by recorded studies.
  $diagnosticName=$enabledDiagnostics[0].Name
  $diagnosticOption='-D'+$enabledDiagnostics[0].Option+'=ON'
  $diagnosticBuild=Join-Path $root "build\$diagnosticName"
  cmake -S $root -B $diagnosticBuild -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_CUDA_ARCHITECTURES=$Architecture" -DCMAKE_BUILD_TYPE=Release $diagnosticOption
  if($LASTEXITCODE){throw 'Diagnostic configure failed'}
  cmake --build $diagnosticBuild --target "synaptic-$diagnosticName" --parallel 4
  if($LASTEXITCODE){throw 'Diagnostic build failed'}
  return
}
cmake -S $root -B "$root\build" -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_CUDA_ARCHITECTURES=$Architecture" -DCMAKE_BUILD_TYPE=Release
if($LASTEXITCODE){throw 'Configure failed'}
cmake --build "$root\build" --parallel 4
if($LASTEXITCODE){throw 'Build failed'}
if(!$SkipTests){
  ctest --test-dir "$root\build" --output-on-failure
  if($LASTEXITCODE){throw 'Native tests failed'}
}
