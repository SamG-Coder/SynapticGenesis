param([string]$Architecture='120', [switch]$SkipTests, [switch]$TraceDiagnostic)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio C++ tools are required.' }
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$ninja=Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
if($TraceDiagnostic){
  # A separate output tree preserves the executable used by recorded studies.
  $traceBuild=Join-Path $root 'build\trace-diagnostic'
  cmake -S $root -B $traceBuild -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_CUDA_ARCHITECTURES=$Architecture" -DCMAKE_BUILD_TYPE=Release -DSG_BUILD_TRACE_DIAGNOSTIC=ON
  if($LASTEXITCODE){throw 'Diagnostic configure failed'}
  cmake --build $traceBuild --target synaptic-trace-diagnostic --parallel 4
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
