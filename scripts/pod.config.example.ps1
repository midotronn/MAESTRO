# AgentLODGE pod connection (Windows). Copy to pod.config.ps1 and edit for your RunPod.
# pod.config.ps1 is gitignored so your host/port never gets committed.
$env:AGENTLODGE_POD_HOST = "213.173.107.238"
$env:AGENTLODGE_POD_PORT = "20642"
$env:AGENTLODGE_POD_KEY  = "$HOME\.ssh\id_ed25519"
$env:AGENTLODGE_POD_WS   = "/workspace"
$env:AGENTLODGE_POD_USER = "root"
# $env:AGENTLODGE_TORCH_INDEX = "cu128"   # or "cpu" for a render-only box
