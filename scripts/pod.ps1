# AgentLODGE pod helper (Windows PowerShell). Makes switching to a fresh RunPod one command.
#
# 1) Point at your pod (either set env vars, or copy pod.config.example.ps1 -> pod.config.ps1 and edit):
#      $env:AGENTLODGE_POD_HOST = "213.173.107.238"
#      $env:AGENTLODGE_POD_PORT = "20642"
#      $env:AGENTLODGE_POD_KEY  = "$HOME\.ssh\id_ed25519"
#      $env:AGENTLODGE_POD_WS   = "/workspace"          # persistent volume root
#      $env:AGENTLODGE_POD_USER = "root"
# 2) Provision everything on the pod (idempotent; re-run after each restart):
#      .\scripts\pod.ps1 setup
# 3) Build a real candidate bank for a song (K seeded LODGE/EDGE takes) and pull it into the app:
#      .\scripts\pod.ps1 bank trs 4
# 4) Run the editor:  uvicorn server.app:app   ->  http://127.0.0.1:8000
#
# Other:  .\scripts\pod.ps1 ssh "nvidia-smi"      run a command on the pod
#         .\scripts\pod.ps1 push <local> <remote> / pull <remote> <local>
param(
  [Parameter(Mandatory = $true)][string]$Command,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
)
$ErrorActionPreference = "Stop"
$cfg = Join-Path $PSScriptRoot "pod.config.ps1"
if (Test-Path $cfg) { . $cfg }
$HostName = $env:AGENTLODGE_POD_HOST
$Port = if ($env:AGENTLODGE_POD_PORT) { $env:AGENTLODGE_POD_PORT } else { "22" }
$Key = if ($env:AGENTLODGE_POD_KEY) { $env:AGENTLODGE_POD_KEY } else { "$HOME\.ssh\id_ed25519" }
$WS = if ($env:AGENTLODGE_POD_WS) { $env:AGENTLODGE_POD_WS } else { "/workspace" }
$User = if ($env:AGENTLODGE_POD_USER) { $env:AGENTLODGE_POD_USER } else { "root" }
if (-not $HostName) { throw "Set AGENTLODGE_POD_HOST (and _PORT/_KEY) or create scripts\pod.config.ps1" }
$Repo = Split-Path $PSScriptRoot -Parent
$Target = "$User@$HostName"

function Pod-SSH([string]$cmd) {
  $output = & ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 `
    -p $Port -i $Key $Target $cmd
  if ($LASTEXITCODE -ne 0) { throw "pod SSH command failed with exit code $LASTEXITCODE" }
  return $output
}
function Pod-Push([string]$local, [string]$remote) {
  & scp -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 `
    -P $Port -i $Key $local "${Target}:$remote"
  if ($LASTEXITCODE -ne 0) { throw "pod upload failed with exit code $LASTEXITCODE" }
}
function Pod-Pull([string]$remote, [string]$local) {
  & scp -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 `
    -P $Port -i $Key "${Target}:$remote" $local
  if ($LASTEXITCODE -ne 0) { throw "pod download failed with exit code $LASTEXITCODE" }
}

switch ($Command) {
  "ssh"  { Pod-SSH ($Args -join " ") }
  "push" { Pod-Push $Args[0] $Args[1] }
  "pull" { Pod-Pull $Args[0] $Args[1] }
  "setup" {
    $torch = if ($env:AGENTLODGE_TORCH_INDEX) { $env:AGENTLODGE_TORCH_INDEX } else { "cu128" }
    $state = Pod-SSH "if [ -d '$WS/AgentLODGE/.git' ] && [ -f '$WS/.maestro_gen_pod_ready' ]; then echo EXISTING_WORKSPACE; else echo FRESH_WORKSPACE; fi"
    if (($state -join "`n") -match "FRESH_WORKSPACE") {
      Write-Host "Fresh workspace detected; uploading the full generation + rendering bootstrap..." -ForegroundColor Cyan
      Pod-Push "$Repo\scripts\setup_gen_pod.sh" "$WS/setup_gen_pod.sh"
      Pod-SSH "sed -i 's/\r`$//' '$WS/setup_gen_pod.sh' && WORKSPACE=$WS AGENTLODGE_TORCH_INDEX=$torch bash '$WS/setup_gen_pod.sh'"
      Pod-SSH "cp '$WS/setup_gen_pod.sh' '$WS/AgentLODGE/scripts/setup_gen_pod.sh'"
    } else {
      Write-Host "Uploading restart setup scripts..." -ForegroundColor Cyan
      Pod-Push "$Repo\scripts\setup_pod.sh" "$WS/AgentLODGE/scripts/setup_pod.sh"
      Pod-Push "$Repo\scripts\build_window_bank.py" "$WS/AgentLODGE/scripts/build_window_bank.py"
      Write-Host "Provisioning pod (system libs + venv + pytorch3d)... this takes a while." -ForegroundColor Cyan
      Pod-SSH "cd $WS/AgentLODGE && sed -i 's/\r`$//' scripts/setup_pod.sh scripts/build_window_bank.py && WORKSPACE=$WS TORCH_INDEX=$torch bash scripts/setup_pod.sh"
    }
  }
  "bank" {
    $sid = $Args[0]; $k = if ($Args[1]) { $Args[1] } else { "4" }
    if (-not $sid) { throw "usage: pod.ps1 bank <sid> [K]" }
    Write-Host "Generating $k real LODGE/EDGE takes for '$sid' on the pod (this is the real work)..." -ForegroundColor Cyan
    Pod-SSH "cd $WS/AgentLODGE && WORKSPACE=$WS AGENTLODGE_BANK_K=$k $WS/AgentLODGE/.venv/bin/python scripts/build_window_bank.py $sid"
    $dst = "$Repo\server\media\$sid\bank"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Write-Host "Pulling bank into $dst ..." -ForegroundColor Cyan
    Pod-Pull "$WS/bank_${sid}_*.npy" "$dst\"
    Get-ChildItem $dst -Filter "bank_${sid}_*.npy" | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
    Write-Host "Done. Restart the server (or reopen the session) to use the richer bank." -ForegroundColor Green
  }
  default { throw "unknown command '$Command' (use: setup | bank | ssh | push | pull)" }
}
