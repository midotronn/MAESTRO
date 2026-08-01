<#
  Host the MAESTRO editor on a RunPod pod and point the project page at it. Makes switching pods a
  one-command operation: syncs code + song data, launches the editor on port 8888 (RunPod's public
  HTTPS proxy port), renders on the pod itself, protects the URL with a password, and rewrites
  docs/static/data/config.json (so the GitHub Pages "Launch editor" button follows the new pod).

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

function PodSSH([string]$cmd, [int]$TimeoutMsg = 0) {
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PodPort -i $PodKey $Target $cmd
}

Write-Host "1/5  Packing code + song data..." -ForegroundColor Cyan
$tar = Join-Path $env:TEMP "maestro_sync.tgz"
if (Test-Path $tar) { Remove-Item $tar -Force }
Push-Location $Repo
try {
  tar czf $tar `
    --exclude="__pycache__" --exclude="*.pyc" --exclude=".git" --exclude=".venv" `
    --exclude="server/sessions" --exclude="server/media/*/_render*" --exclude="server/media/*/_cmp*" `
    --exclude="docs/static/videos" `
    server agentlodge scripts docs requirements.txt README.md
  if ($LASTEXITCODE -ne 0) { throw "tar failed" }
} finally { Pop-Location }
"{0:N1} MB packed" -f ((Get-Item $tar).Length / 1MB)

Write-Host "2/5  Uploading to the pod..." -ForegroundColor Cyan
scp -o StrictHostKeyChecking=no -P $PodPort -i $PodKey $tar "${Target}:$WS/maestro_sync.tgz"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

Write-Host "3/5  Extracting + launching the editor on :8888..." -ForegroundColor Cyan
PodSSH "mkdir -p $WS/AgentLODGE && tar xzf $WS/maestro_sync.tgz -C $WS/AgentLODGE && sed -i 's/\r$//' $WS/AgentLODGE/scripts/serve_on_pod.sh" | Out-Null
$launch = "OPENAI_API_KEY='$OpenAIKey' MAESTRO_AUTH_USER='$AuthUser' MAESTRO_AUTH_PASS='$AuthPass' WORKSPACE='$WS' bash $WS/AgentLODGE/scripts/serve_on_pod.sh"
$out = PodSSH $launch
$out | ForEach-Object { Write-Host "    $_" }
if (-not (($out -join "`n") -match "MAESTRO_EDITOR_UP")) { throw "editor did not start on the pod (see log above)" }

Write-Host "4/5  Verifying the public URL..." -ForegroundColor Cyan
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

Write-Host "5/5  Pointing the project page at this pod..." -ForegroundColor Cyan
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
