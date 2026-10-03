param([string]$Architecture='120', [switch]$SkipTests, [switch]$TraceDiagnostic, [switch]$AdaptationProbe)
$ErrorActionPreference='Stop'
if($TraceDiagnostic -and $AdaptationProbe){throw 'Select one diagnostic build per invocation.'}
$root=$PSScriptRoot
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio C++ tools are required.' }
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$ninja=Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
if($TraceDiagnostic -or $AdaptationProbe){
  # A separate output tree preserves the executable used by recorded studies.
  $diagnosticName=if($TraceDiagnostic){'trace-diagnostic'}else{'adaptation-probe'}
  $diagnosticOption=if($TraceDiagnostic){'-DSG_BUILD_TRACE_DIAGNOSTIC=ON'}else{'-DSG_BUILD_ADAPTATION_PROBE=ON'}
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
