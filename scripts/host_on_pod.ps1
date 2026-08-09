<#
  Host the MAESTRO editor on a RunPod pod and point the project page at it. Makes switching pods a
  one-command operation: validates and syncs code + the exact audited motion-bank assets, launches
  the editor on port 8888 (RunPod's public HTTPS proxy port), renders on the pod itself, protects the
  URL with a password, and rewrites docs/static/data/config.json (so the GitHub Pages "Launch
  editor" button follows the new pod).

  Usage:
    .\scripts\host_on_pod.ps1 -PodHost 213.173.109.111 -PodPort 41920 -PodId 2mliggkea7jgtt
    .\scripts\host_on_pod.ps1 -PodHost <ip> -PodPort <port> -PodId <id> -AuthPass "mypw" -NoPush

  Required: -PodHost -PodPort -PodId (all from the RunPod dashboard / pod env RUNPOD_POD_ID).
  Optional: -PodKey (default ~/.ssh/id_ed25519), -OpenAIKey (default $env:OPENAI_API_KEY),
            -AuthUser (default "maestro"), -AuthPass (generated if omitted), -NoPush.
#>
param(
  [Parameter(Mandatory = $true)][string]$PodHost,
  [Parameter(Mandatory = $true)][string]$PodPort,
  [Parameter(Mandatory = $true)][string]$PodId,
  [string]$PodKey = "$HOME\.ssh\id_ed25519",
  [string]$OpenAIKey = $env:OPENAI_API_KEY,
  [string]$AuthUser = "maestro",
  [string]$AuthPass = "",
  [string]$User = "root",
  [string]$WS = "/workspace",
  [switch]$NoPush
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
$Target = "$User@$PodHost"
$ProxyUrl = "https://$PodId-8888.proxy.runpod.net/"
if (-not (Test-Path $PodKey)) { throw "SSH key not found: $PodKey" }
if (-not $AuthPass) {
  $AuthPass = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 20 | ForEach-Object { [char]$_ })
  Write-Host "Generated editor password: $AuthPass" -ForegroundColor Yellow
}

function PodSSH([string]$cmd) {
  $output = & ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 `
    -p $PodPort -i $PodKey $Target $cmd
  if ($LASTEXITCODE -ne 0) { throw "pod command failed" }
  return $output
}

function PodSSHScript([string]$script) {
  $output = $script | & ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 `
    -p $PodPort -i $PodKey $Target "bash -s"
  if ($LASTEXITCODE -ne 0) { throw "pod deployment command failed" }
  return $output
}

Write-Host "1/6  Validating the required motion audit receipt..." -ForegroundColor Cyan
$receipt = Join-Path $Repo "assets\motion_bank\audit_receipt.json"
if (-not (Test-Path $receipt)) {
  throw "required passing motion audit receipt is missing: $receipt"
}
$localPython = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $localPython)) { $localPython = "python" }
Push-Location $Repo
try {
  & $localPython -c "from agentlodge.editor.motion_audit import validate_audit_receipt; validate_audit_receipt()"
  if ($LASTEXITCODE -ne 0) { throw "motion audit receipt is absent, stale, invalid, or not passing" }
} finally { Pop-Location }
Write-Host "    passing receipt validated locally." -ForegroundColor Green

Write-Host "2/6  Packing code + exact motion-bank assets..." -ForegroundColor Cyan
$tar = Join-Path $Repo ".maestro_sync.tgz"
if (Test-Path $tar) { Remove-Item $tar -Force }
Push-Location $Repo
try {
  tar czf $tar `
    --exclude="__pycache__" --exclude="*.pyc" --exclude=".git" --exclude=".venv" `
    --exclude="server/sessions" --exclude="server/media/*/_render*" --exclude="server/media/*/_cmp*" `
    --exclude="docs/static/videos" `
    server agentlodge scripts docs assets/motion_bank requirements.txt README.md
  if ($LASTEXITCODE -ne 0) { throw "tar failed" }
} finally { Pop-Location }
"{0:N1} MB packed" -f ((Get-Item $tar).Length / 1MB)

Write-Host "3/6  Uploading, staging, and validating on the pod..." -ForegroundColor Cyan
try {
  scp -o StrictHostKeyChecking=no -P $PodPort -i $PodKey $tar "${Target}:$WS/maestro_sync.tgz"
  if ($LASTEXITCODE -ne 0) { throw "scp failed" }
} finally {
  if (Test-Path $tar) { Remove-Item $tar -Force }
}

$deployScript = @'
set -euo pipefail
workspace="__WORKSPACE__"
live="$workspace/AgentLODGE"
stage="$workspace/maestro_stage"
archive="$workspace/maestro_sync.tgz"
backup="$workspace/.motion_bank.previous.$$"
trap 'rm -rf "$stage" "$backup"; rm -f "$archive"' EXIT

rm -rf "$stage"
mkdir -p "$stage"
tar xzf "$archive" -C "$stage"
sed -i 's/\r$//' "$stage/scripts/serve_on_pod.sh"
test -f "$stage/assets/motion_bank/audit_receipt.json"
cd "$stage"
PYTHONPATH="$stage" "$live/.venv/bin/python" -c 'from pathlib import Path; from agentlodge.editor.motion_audit import validate_audit_receipt; validate_audit_receipt(Path("assets/motion_bank/audit_receipt.json"), root=Path(".")); print("MOTION_AUDIT_RECEIPT_OK")'

mkdir -p "$live/assets"
if [ -e "$live/assets/motion_bank" ]; then
  mv "$live/assets/motion_bank" "$backup"
fi
if ! mv "$stage/assets/motion_bank" "$live/assets/motion_bank"; then
  [ ! -e "$backup" ] || mv "$backup" "$live/assets/motion_bank"
  exit 1
fi
rm -rf "$backup" "$stage/assets"
tar -C "$stage" -cf - . | tar -C "$live" -xf -
'@.Replace("__WORKSPACE__", $WS)
$deployOut = PodSSHScript $deployScript
if (-not (($deployOut -join "`n") -match "MOTION_AUDIT_RECEIPT_OK")) {
  throw "staged motion audit validation did not complete"
}
Write-Host "    staged receipt passed; exact assets/motion_bank tree replaced." -ForegroundColor Green

Write-Host "4/6  Launching the editor on :8888..." -ForegroundColor Cyan
$openAI64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($OpenAIKey))
$authUser64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($AuthUser))
$authPass64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($AuthPass))
$launchScript = @'
set -e
export OPENAI_API_KEY="$(printf '%s' '__OPENAI64__' | base64 -d)"
export MAESTRO_AUTH_USER="$(printf '%s' '__AUTHUSER64__' | base64 -d)"
export MAESTRO_AUTH_PASS="$(printf '%s' '__AUTHPASS64__' | base64 -d)"
export WORKSPACE="__WORKSPACE__"
bash "$WORKSPACE/AgentLODGE/scripts/serve_on_pod.sh"
'@
$launchScript = $launchScript.Replace("__OPENAI64__", $openAI64)
$launchScript = $launchScript.Replace("__AUTHUSER64__", $authUser64)
$launchScript = $launchScript.Replace("__AUTHPASS64__", $authPass64)
$launchScript = $launchScript.Replace("__WORKSPACE__", $WS)
$out = PodSSHScript $launchScript
$launchScript = $null
$openAI64 = $null
$authUser64 = $null
$authPass64 = $null
$out | ForEach-Object { Write-Host "    $_" }
if (-not (($out -join "`n") -match "MAESTRO_EDITOR_UP")) { throw "editor did not start on the pod (see log above)" }

Write-Host "5/6  Verifying the public URL..." -ForegroundColor Cyan
$pair = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${AuthUser}:${AuthPass}"))
$ok = $false
for ($i = 0; $i -lt 6; $i++) {
  Start-Sleep 5
  try {
    $r = Invoke-WebRequest -UseBasicParsing $ProxyUrl -Headers @{ Authorization = "Basic $pair" } -TimeoutSec 25
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch { }
}
if ($ok) { Write-Host "    reachable: $ProxyUrl" -ForegroundColor Green }
else { Write-Host "    WARNING: not reachable yet (the RunPod proxy can take a minute)." -ForegroundColor Yellow }

Write-Host "6/6  Pointing the project page at this pod..." -ForegroundColor Cyan
$cfgPath = Join-Path $Repo "docs\static\data\config.json"
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
$cfg.editorUrl = $ProxyUrl
($cfg | ConvertTo-Json -Depth 5) | Set-Content $cfgPath -Encoding utf8
if ($NoPush) {
  Write-Host "    updated config.json (skipped git push; -NoPush)." -ForegroundColor Yellow
} else {
  Push-Location $Repo
  try {
    git add docs/static/data/config.json
    git commit -m "Point project page at pod $PodId" | Out-Null
    git push origin interactive-editor | Out-Null
    Write-Host "    committed + pushed; GitHub Pages will update in ~1 min." -ForegroundColor Green
  } finally { Pop-Location }
}

Write-Host ""
Write-Host "DONE. MAESTRO editor is live:" -ForegroundColor Green
Write-Host "  URL:      $ProxyUrl"
Write-Host "  Login:    $AuthUser / $AuthPass"
Write-Host "  Restart:  re-run this same command (idempotent)."
