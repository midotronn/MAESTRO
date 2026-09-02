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
#    Provision and start the exact warm four-GPU SLA stack:
#      .\scripts\pod.ps1 setup4
#    setup4 also replaces the public RunPod service on port 8888 with the interview-ready editor.
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
$GitRefExitCode = 0
$GitRef = if ($env:AGENTLODGE_GIT_REF) {
  $env:AGENTLODGE_GIT_REF
} else {
  $value = & git -C $Repo branch --show-current
  $GitRefExitCode = $LASTEXITCODE
  $value
}
$GitRef = "$GitRef".Trim()
if (-not $GitRef) {
  $GitRef = & git -C $Repo rev-parse HEAD
  $GitRefExitCode = $LASTEXITCODE
  $GitRef = "$GitRef".Trim()
}
if ($GitRefExitCode -ne 0 -or $GitRef -notmatch '^[A-Za-z0-9._/-]+$') {
  throw "Could not determine a safe AGENTLODGE_GIT_REF"
}
$GitUrl = if ($env:AGENTLODGE_GIT_URL) {
  $env:AGENTLODGE_GIT_URL
} else {
  "https://github.com/midotronn/MAESTRO.git"
}

function Pod-SSH([string]$cmd) {
  $cmd = $cmd.Replace("`r", "")
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
      Pod-Push "$Repo\scripts\build_egl_selector.sh" "$WS/build_egl_selector.sh"
      Pod-Push "$Repo\scripts\egl_cuda_device_selector.c" "$WS/egl_cuda_device_selector.c"
      Pod-SSH "sed -i 's/\r`$//' '$WS/setup_gen_pod.sh' && WORKSPACE=$WS AGENTLODGE_GIT_REF='$GitRef' AGENTLODGE_TORCH_INDEX=$torch bash '$WS/setup_gen_pod.sh'"
      Pod-SSH "cp '$WS/setup_gen_pod.sh' '$WS/AgentLODGE/scripts/setup_gen_pod.sh'"
    } else {
      Write-Host "Uploading restart setup scripts..." -ForegroundColor Cyan
      Pod-Push "$Repo\scripts\setup_pod.sh" "$WS/AgentLODGE/scripts/setup_pod.sh"
      Pod-Push "$Repo\scripts\build_egl_selector.sh" "$WS/AgentLODGE/scripts/build_egl_selector.sh"
      Pod-Push "$Repo\scripts\egl_cuda_device_selector.c" "$WS/AgentLODGE/scripts/egl_cuda_device_selector.c"
      Pod-Push "$Repo\scripts\build_window_bank.py" "$WS/AgentLODGE/scripts/build_window_bank.py"
      Write-Host "Provisioning pod (system libs + venv + pytorch3d)... this takes a while." -ForegroundColor Cyan
      Pod-SSH "cd $WS/AgentLODGE && sed -i 's/\r`$//' scripts/setup_pod.sh scripts/build_egl_selector.sh scripts/build_window_bank.py && WORKSPACE=$WS TORCH_INDEX=$torch bash scripts/setup_pod.sh"
    }
  }
  "setup4" {
    $torch = if ($env:AGENTLODGE_TORCH_INDEX) { $env:AGENTLODGE_TORCH_INDEX } else { "cu128" }
    $publicPort = if ($env:AGENTLODGE_PUBLIC_PORT) { $env:AGENTLODGE_PUBLIC_PORT } else { "8888" }
    if ($publicPort -notmatch '^\d+$') {
      throw "AGENTLODGE_PUBLIC_PORT must be numeric"
    }
    $remotePlannerKey = ""
    $requirePlanner = "0"
    if ($env:AGENTLODGE_OAI_KEY_FILE) {
      $plannerKey = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
        $env:AGENTLODGE_OAI_KEY_FILE
      )
      if (-not (Test-Path -LiteralPath $plannerKey -PathType Leaf)) {
        throw "Planner key file not found: $plannerKey"
      }
      Write-Host "Provisioning the planner credential from a private local file..." -ForegroundColor Cyan
      Pod-Push $plannerKey "$WS/.oai_key.upload"
      try {
        Pod-SSH "set -e; trap 'rm -f $WS/.oai_key.upload' EXIT; umask 077; install -m 600 '$WS/.oai_key.upload' /root/.oai_key; test -s /root/.oai_key; test `"`$(stat -c %a /root/.oai_key)`" = 600"
      } catch {
        Pod-SSH "rm -f '$WS/.oai_key.upload'" | Out-Null
        throw
      }
      $remotePlannerKey = "/root/.oai_key"
      $requirePlanner = "1"
    }
    Write-Host "Provisioning the exact warm four-GPU MAESTRO stack from '$GitRef'..." -ForegroundColor Cyan
    Pod-SSH @"
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates git
if [ ! -d '$WS/AgentLODGE/.git' ]; then
  if git ls-remote --exit-code --heads '$GitUrl' 'refs/heads/$GitRef' >/dev/null 2>&1; then
    git clone --branch '$GitRef' --single-branch '$GitUrl' '$WS/AgentLODGE'
  else
    git clone '$GitUrl' '$WS/AgentLODGE'
    git -C '$WS/AgentLODGE' fetch --depth=1 origin '$GitRef'
    git -C '$WS/AgentLODGE' checkout --detach FETCH_HEAD
  fi
else
  if ! git -C '$WS/AgentLODGE' diff --quiet ||
      ! git -C '$WS/AgentLODGE' diff --cached --quiet; then
    echo 'Refusing to replace a modified pod checkout.' >&2
    exit 1
  fi
  if printf '%s\n' '$GitRef' | grep -Eq '^[0-9a-fA-F]{40}$' &&
      git -C '$WS/AgentLODGE' cat-file -e '$GitRef^{commit}' 2>/dev/null; then
    echo 'Pinned commit is already cached; skipping the remote fetch.'
    git -C '$WS/AgentLODGE' checkout --detach '$GitRef'
  else
    git -C '$WS/AgentLODGE' fetch origin '$GitRef'
    if git -C '$WS/AgentLODGE' show-ref --verify --quiet 'refs/remotes/origin/$GitRef'; then
      git -C '$WS/AgentLODGE' checkout -B '$GitRef' 'origin/$GitRef'
    else
      git -C '$WS/AgentLODGE' checkout --detach FETCH_HEAD
    fi
  fi
fi
cd '$WS/AgentLODGE'
WORKSPACE='$WS' \
AGENTLODGE_GIT_URL='$GitUrl' \
AGENTLODGE_GIT_REF='$GitRef' \
AGENTLODGE_TORCH_INDEX='$torch' \
AGENTLODGE_PUBLIC_PORT='$publicPort' \
MAESTRO_PUBLIC_INTERVIEW_MODE='1' \
OAI_KEY_FILE='$remotePlannerKey' \
AGENTLODGE_REQUIRE_LLM_PLANNER='$requirePlanner' \
bash scripts/setup_four_gpu_pod.sh
"@
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
  default { throw "unknown command '$Command' (use: setup4 | setup | bank | ssh | push | pull)" }
}
