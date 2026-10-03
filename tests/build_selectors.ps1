param(
  [string]$BuildScript=(Join-Path $PSScriptRoot '../build.ps1'),
  [Parameter(Mandatory=$true)][string]$Report
)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $Report){throw 'Use a fresh build-selector report.'}
$scriptText=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $BuildScript))
$parseTokens=$null
$parseErrors=$null
[System.Management.Automation.Language.Parser]::ParseInput($scriptText,[ref]$parseTokens,[ref]$parseErrors) | Out-Null
if($parseErrors.Count){throw 'Build script has PowerShell parse errors.'}
$boundary=$scriptText.IndexOf('$root=$PSScriptRoot')
if($boundary -lt 0){throw 'Cannot isolate selection before toolchain setup.'}
# Evaluate only parameter binding and selection; do not enter toolchain setup.
$selector=[scriptblock]::Create($scriptText.Substring(0,$boundary)+"`n"+'$enabledDiagnostics')
$expected=@{
  TraceDiagnostic=@{Name='trace-diagnostic'; Option='SG_BUILD_TRACE_DIAGNOSTIC'}
  AdaptationProbe=@{Name='adaptation-probe'; Option='SG_BUILD_ADAPTATION_PROBE'}
  ReplayPriorityProbe=@{Name='replay-priority-probe'; Option='SG_BUILD_REPLAY_PRIORITY_PROBE'}
  ReplaySelectionProbe=@{Name='replay-selection-probe'; Option='SG_BUILD_REPLAY_SELECTION_PROBE'}
  CapacityProbe=@{Name='capacity-probe'; Option='SG_BUILD_CAPACITY_PROBE'}
  ViewAllocationProbe=@{Name='view-allocation-probe'; Option='SG_BUILD_VIEW_ALLOCATION_PROBE'}
  EarlyLearningProbe=@{Name='early-learning-probe'; Option='SG_BUILD_EARLY_LEARNING_PROBE'}
}
if(@(& $selector).Count -ne 0){throw 'Default build selected a diagnostic.'}
$cases=@()
$keys=@($expected.Keys | Sort-Object)
foreach($key in $keys){
  $switches=@{$key=$true}
  $actual=@(& $selector @switches)
  if($actual.Count -ne 1 -or $actual[0].Name -ne $expected[$key].Name -or $actual[0].Option -ne $expected[$key].Option){
    throw "Incorrect target selection for $key"
  }
  $cases+=@{switch=$key; name=$actual[0].Name; option=$actual[0].Option}
}
$rejected=0
for($i=0;$i -lt $keys.Count;$i++){
  for($j=$i+1;$j -lt $keys.Count;$j++){
    $switches=@{$keys[$i]=$true; $keys[$j]=$true}
    $refused=$false
    try{& $selector @switches | Out-Null}
    catch{
      if($_.Exception.Message -ne 'Select one diagnostic build per invocation.'){throw}
      $refused=$true
    }
    if(!$refused){throw 'Two diagnostic selectors were accepted.'}
    $rejected++
  }
}
$result=@{
  passed=$true
  script_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $BuildScript).Hash.ToLowerInvariant()
  no_selection_uses_normal_build=$true
  selectors=$cases
  conflicting_pairs_rejected=$rejected
  toolchain_or_cuda_work_requested=$false
}
$reportPath=[IO.Path]::GetFullPath($Report)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($reportPath)) | Out-Null
$json=($result | ConvertTo-Json -Depth 8).Replace("`r`n","`n")+"`n"
[IO.File]::WriteAllText($reportPath,$json,[Text.UTF8Encoding]::new($false))
Write-Output "PASS: $($cases.Count) selectors, $rejected conflicting pairs; no toolchain or CUDA work."
