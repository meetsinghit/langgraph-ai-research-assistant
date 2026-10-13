<#
.SYNOPSIS
  Core launcher logic for the AI Research Assistant (Phase 4D.3).

  Invoked by the thin wrappers:
    start-research-assistant.bat       production (serves the built UI)
    start-research-assistant-dev.bat   development (Vite dev server)
    stop-research-assistant.bat        stops only what we started

  Design rules:
  - never print secrets (the access token is offered via the
    clipboard only, on an explicit keypress);
  - never kill a process we did not start (PID files + command-line
    marker verification before taskkill);
  - duplicate instances are detected via the health endpoint;
  - SQLite data in data/ is untouched by start/stop;
  - no administrator privileges required.
#>
param(
    [ValidateSet("start", "dev", "stop", "selftest")]
    [string]$Action = "start",
    [string]$Command = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root "ai_poc\Scripts\python.exe"
$DataDir = Join-Path $Root "data"
$LogDir = Join-Path $Root "logs"
$EnvFile = Join-Path $Root ".env"
$BackendPidFile = Join-Path $DataDir "backend.pid"
$FrontendPidFile = Join-Path $DataDir "frontend.pid"
$BackendMarker = "app.main"
$FrontendMarker = "vite"

function Write-Info([string]$Message) { Write-Host $Message }
function Write-Step([string]$Message) { Write-Host ("==> " + $Message) -ForegroundColor Cyan }
function Fail([string]$Message) {
    Write-Host ("ERROR: " + $Message) -ForegroundColor Red
    exit 1
}

# --- pure helpers (covered by selftest + pytest) -----------------------------

function Get-EnvFileValue {
    param([string]$Path, [string]$Key)
    if (-not (Test-Path $Path)) { return $null }
    foreach ($line in Get-Content $Path) {
        $t = $line.Trim()
        if ($t -eq "" -or $t.StartsWith("#")) { continue }
        $idx = $t.IndexOf("=")
        if ($idx -lt 1) { continue }
        if ($t.Substring(0, $idx).Trim().ToUpperInvariant() -ne $Key.ToUpperInvariant()) { continue }
        $v = $t.Substring($idx + 1).Trim()
        if ($v.Length -ge 2 -and (($v[0] -eq '"' -and $v[$v.Length - 1] -eq '"') -or ($v[0] -eq "'" -and $v[$v.Length - 1] -eq "'"))) {
            $v = $v.Substring(1, $v.Length - 2)
        }
        return $v
    }
    return $null
}

function Test-PortInUse {
    param([int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $ar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $reached = $ar.AsyncWaitHandle.WaitOne(400, $false)
        return ($reached -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Get-HttpOk {
    param([string]$Url, [int]$TimeoutSecs = 2)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSecs
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Stop-AppProcess {
    # Stops ONLY a process we recorded whose command line still
    # carries our marker; anything else is refused untouched.
    param([string]$PidFile, [string]$Marker)
    if (-not (Test-Path $PidFile)) { return "no-pidfile" }
    $procId = 0
    try { $procId = [int]((Get-Content $PidFile -TotalCount 1).Trim()) } catch { $procId = 0 }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    if ($procId -le 0) { return "bad-pidfile" }
    $proc = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $procId) -ErrorAction SilentlyContinue
    if (-not $proc) { return "not-running" }
    $cmd = ""
    if ($proc.CommandLine) { $cmd = $proc.CommandLine }
    if ($cmd.IndexOf($Marker) -lt 0) { return "refused-foreign" }
    & taskkill /PID $procId /T /F 2>$null | Out-Null
    return "stopped"
}

# --- shared startup steps ----------------------------------------------------

function Assert-Venv {
    if (-not (Test-Path $VenvPython)) {
        Fail ("Python virtual environment not found at ai_poc\.`n" +
            "Fix: install Python 3.12+, then from this folder run:`n" +
            "    python -m venv ai_poc`n" +
            "    ai_poc\Scripts\python.exe -m pip install -r requirements.txt")
    }
    & $VenvPython -c "import fastapi, uvicorn, langgraph, pydantic_settings" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Fail ("Backend dependencies are missing or broken.`n" +
            "Fix: ai_poc\Scripts\python.exe -m pip install -r requirements.txt")
    }
}

function Get-ApiPort {
    $value = Get-EnvFileValue -Path $EnvFile -Key "API_PORT"
    if ($value) {
        $port = 0
        if ([int]::TryParse($value, [ref]$port) -and $port -gt 0 -and $port -lt 65536) { return $port }
        Fail ("API_PORT in .env is not a valid port: " + $value)
    }
    return 8000
}

function Assert-PortFreeOrOurs {
    param([int]$Port)
    if (Get-HttpOk ("http://127.0.0.1:" + $Port + "/api/health")) {
        return "already-running"
    }
    if (Test-PortInUse -Port $Port) {
        Fail ("Port " + $Port + " is used by another program, so the assistant cannot start.`n" +
            "Fix: close that program, or set API_PORT=<free port> in .env and retry.")
    }
    return "free"
}

function Warn-AboutProviderKey {
    $key = Get-EnvFileValue -Path $EnvFile -Key "GEMINI_API_KEY"
    if (-not $key) {
        Write-Host ("WARNING: no GEMINI_API_KEY found in .env -- the app will start," +
            " but research runs will report a provider error until you add one.") -ForegroundColor Yellow
    }
}

function Wait-ForHealth {
    param([int]$Port, [System.Diagnostics.Process]$Process)
    for ($i = 0; $i -lt 40; $i++) {
        if (Get-HttpOk ("http://127.0.0.1:" + $Port + "/api/health")) { return $true }
        if ($Process -and $Process.HasExited) { return $false }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Show-StartupFailureHelp {
    param([int]$Port)
    Write-Host "The backend did not become healthy in time. Log tail:" -ForegroundColor Red
    $errLog = Join-Path $LogDir "server-error.log"
    $outLog = Join-Path $LogDir "server.log"
    foreach ($log in @($errLog, $outLog)) {
        if (Test-Path $log) {
            Write-Host ("--- " + $log + " ---")
            Get-Content $log -Tail 20
        }
    }
    Write-Host ("Common fixes: ensure .env has a valid GEMINI_API_KEY; if " + $Port + " is busy, set API_PORT in .env; see docs\\OPERATIONS.md.")
}

function Offer-TokenClipboard {
    # The token is NEVER printed. It can be copied to the clipboard
    # on an explicit keypress, which is how the browser dialog is fed.
    if ($env:RESEARCH_ASSISTANT_UNATTENDED -eq "1") { return }
    $tokenFile = Join-Path $DataDir "app_token.txt"
    Write-Info ""
    Write-Info ("Access token file: " + $tokenFile + " (never shown on screen)")
    Write-Info "The browser will ask for the access token on first launch."
    if (-not (Test-Path $tokenFile)) { return }
    $answer = Read-Host "Press C then Enter to copy the access token to the clipboard (or just Enter to skip)"
    if ($answer -match "^[Cc]$") {
        try {
            Get-Content $tokenFile -TotalCount 1 | Set-Clipboard
            Write-Info "Copied. Paste it into the browser's connection dialog (Ctrl+V)."
        } catch {
            Write-Host "Could not access the clipboard -- copy the token manually from the file above." -ForegroundColor Yellow
        }
    }
}

function Open-Browser {
    param([string]$Url)
    if ($env:RESEARCH_ASSISTANT_NO_BROWSER -eq "1") { return }
    Start-Process $Url | Out-Null
}

function Start-Backend {
    param([int]$Port)
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
    $outLog = Join-Path $LogDir "server.log"
    $errLog = Join-Path $LogDir "server-error.log"
    $process = Start-Process -FilePath $VenvPython `
        -ArgumentList "-m", "app.main" `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -PassThru -WindowStyle Hidden
    Set-Content -Path $BackendPidFile -Value $process.Id
    if (-not (Wait-ForHealth -Port $Port -Process $process)) {
        Show-StartupFailureHelp -Port $Port
        Stop-AppProcess -PidFile $BackendPidFile -Marker $BackendMarker | Out-Null
        exit 1
    }
    return $process
}

function Ensure-UiBuilt {
    $index = Join-Path $Root "frontend\dist\index.html"
    if (Test-Path $index) { return }
    Write-Step "UI not built yet -- building it now (one-time, takes a minute)"
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) {
        Fail ("The browser UI is not built and npm is not installed.`n" +
            "Fix: install Node.js 20+, then run:`n" +
            "    cd frontend`n    npm install`n    npm run build`n" +
            "and start again. (Development alternative: start-research-assistant-dev.bat)")
    }
    Push-Location (Join-Path $Root "frontend")
    try {
        if (-not (Test-Path "node_modules")) {
            & npm install 2>&1 | Select-Object -Last 5
            if ($LASTEXITCODE -ne 0) { Fail "npm install failed -- see the output above." }
        }
        & npm run build 2>&1 | Select-Object -Last 5
        if ($LASTEXITCODE -ne 0) { Fail "UI build failed -- see the output above." }
    } finally {
        Pop-Location
    }
    if (-not (Test-Path $index)) { Fail "Build finished but frontend\dist\index.html is missing." }
}

# --- actions -----------------------------------------------------------------

function Invoke-Start {
    Write-Step "Starting the AI Research Assistant (production mode)"
    Assert-Venv
    $port = Get-ApiPort
    Ensure-UiBuilt
    Warn-AboutProviderKey

    $state = Assert-PortFreeOrOurs -Port $port
    if ($state -eq "already-running") {
        Write-Info ("Already running on port " + $Port + " -- opening the browser instead (no duplicate started).")
        Open-Browser ("http://127.0.0.1:" + $port + "/")
        exit 0
    }

    Start-Backend -Port $port | Out-Null
    Write-Info ("Backend is healthy on http://127.0.0.1:" + $port + "/")
    Offer-TokenClipboard
    Open-Browser ("http://127.0.0.1:" + $port + "/")
    Write-Info ""
    if ($env:RESEARCH_ASSISTANT_UNATTENDED -eq "1") {
        Write-Info "Unattended mode: the backend keeps running (stop with stop-research-assistant.bat)."
        exit 0
    }
    Write-Info "The app keeps running in the background. Close this window to leave it running,"
    Write-Info "or press Enter to stop it now (stop-research-assistant.bat also works later)."
    Read-Host "Press Enter to stop" | Out-Null
    $result = Stop-AppProcess -PidFile $BackendPidFile -Marker $BackendMarker
    Write-Info ("Stopped (" + $result + "). Your saved research stays in data\.")
}

function Invoke-Dev {
    Write-Step "Starting the AI Research Assistant (development mode)"
    Assert-Venv
    $port = Get-ApiPort
    Warn-AboutProviderKey

    $state = Assert-PortFreeOrOurs -Port $port
    if ($state -eq "already-running") {
        Write-Info ("Backend already running on port " + $port + " -- reusing it.")
    } else {
        Start-Backend -Port $port | Out-Null
        Write-Info ("Backend is healthy on http://127.0.0.1:" + $port + "/")
    }

    if (-not (Test-Path (Join-Path $Root "frontend\node_modules"))) {
        Write-Step "Installing UI dependencies (one-time)"
        Push-Location (Join-Path $Root "frontend")
        try {
            & npm install 2>&1 | Select-Object -Last 5
            if ($LASTEXITCODE -ne 0) { Fail "npm install failed -- see the output above." }
        } finally { Pop-Location }
    }

    Write-Step "Starting the Vite dev server"
    $viteLog = Join-Path $LogDir "vite.log"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $vite = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "npm", "run", "dev" `
        -WorkingDirectory (Join-Path $Root "frontend") `
        -RedirectStandardOutput $viteLog `
        -RedirectStandardError (Join-Path $LogDir "vite-error.log") `
        -PassThru -WindowStyle Hidden
    Set-Content -Path $FrontendPidFile -Value $vite.Id

    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if (Get-HttpOk "http://127.0.0.1:5173/") { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        Write-Host "The dev server did not respond on port 5173. Log tail:" -ForegroundColor Red
        if (Test-Path $viteLog) { Get-Content $viteLog -Tail 20 }
        exit 1
    }
    Offer-TokenClipboard
    Open-Browser "http://127.0.0.1:5173/"
    Write-Info ""
    Write-Info "Dev mode: UI with hot reload on http://127.0.0.1:5173/."
    if ($env:RESEARCH_ASSISTANT_UNATTENDED -eq "1") {
        Write-Info "Unattended mode: both keep running (stop with stop-research-assistant.bat)."
        exit 0
    }
    Write-Info "Close this window to leave both running; press Enter to stop both."
    Read-Host "Press Enter to stop" | Out-Null
    Invoke-Stop
}

function Invoke-Stop {
    Write-Step "Stopping the AI Research Assistant (only processes we started)"
    $backend = Stop-AppProcess -PidFile $BackendPidFile -Marker $BackendMarker
    $frontend = Stop-AppProcess -PidFile $FrontendPidFile -Marker $FrontendMarker
    Write-Info ("backend: " + $backend + "; frontend: " + $frontend)
    Write-Info "Saved research remains in data\. Nothing else was touched."
}

function Invoke-SelfTest {
    param([string]$Name)
    switch ($Name) {
        "Get-EnvValueTest" {
            $tmp = [System.IO.Path]::GetTempFileName()
            Set-Content -Path $tmp -Value ("# comment", "GEMINI_API_KEY=abc-123", "API_PORT=8123", "QUOTED=`"q-v`"")
            $r1 = Get-EnvFileValue -Path $tmp -Key "gemini_api_key"
            $r2 = Get-EnvFileValue -Path $tmp -Key "API_PORT"
            $r3 = Get-EnvFileValue -Path $tmp -Key "NOPE"
            $r4 = Get-EnvFileValue -Path $tmp -Key "QUOTED"
            Remove-Item $tmp -Force
            if ($r1 -eq "abc-123" -and $r2 -eq "8123" -and ($null -eq $r3) -and $r4 -eq "q-v") {
                Write-Output "ok"
            } else {
                Fail ("env parse failed: r1=$r1 r2=$r2 r3=$r3 r4=$r4")
            }
        }
        "Test-PortProbeTest" {
            $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
            $listener.Start()
            $busyPort = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
            $inUse = Test-PortInUse -Port $busyPort
            $listener.Stop()
            Start-Sleep -Milliseconds 150
            $freeInUse = Test-PortInUse -Port $busyPort
            if ($inUse -and -not $freeInUse) { Write-Output "ok" }
            else { Fail "port probe failed: inUse=$inUse freeInUse=$freeInUse" }
        }
        "Test-SafeStopTest" {
            $markerProc = Start-Process powershell -ArgumentList "-NoProfile", "-Command", "Start-Sleep -Seconds 60; # app.main" -PassThru -WindowStyle Hidden
            $foreignProc = Start-Process powershell -ArgumentList "-NoProfile", "-Command", "Start-Sleep -Seconds 60" -PassThru -WindowStyle Hidden
            Start-Sleep -Milliseconds 500
            $markerPidFile = Join-Path $env:TEMP "aipoc-selftest-marker.pid"
            $foreignPidFile = Join-Path $env:TEMP "aipoc-selftest-foreign.pid"
            Set-Content -Path $markerPidFile -Value $markerProc.Id
            Set-Content -Path $foreignPidFile -Value $foreignProc.Id
            $r1 = Stop-AppProcess -PidFile $markerPidFile -Marker $BackendMarker
            $r2 = Stop-AppProcess -PidFile $foreignPidFile -Marker $BackendMarker
            Start-Sleep -Milliseconds 500
            $markerAlive = [bool](Get-Process -Id $markerProc.Id -ErrorAction SilentlyContinue)
            $foreignAlive = [bool](Get-Process -Id $foreignProc.Id -ErrorAction SilentlyContinue)
            if ($markerAlive) { Stop-Process -Id $markerProc.Id -Force -ErrorAction SilentlyContinue }
            if ($foreignAlive) { Stop-Process -Id $foreignProc.Id -Force -ErrorAction SilentlyContinue }
            if ($r1 -eq "stopped" -and -not $markerAlive -and $r2 -eq "refused-foreign" -and $foreignAlive) {
                Write-Output "ok"
            } else {
                Fail ("safe-stop failed: r1=$r1 r2=$r2 markerAlive=$markerAlive foreignAlive=$foreignAlive")
            }
        }
        default { Fail ("unknown selftest: " + $Name) }
    }
}

switch ($Action) {
    "start" { Invoke-Start }
    "dev" { Invoke-Dev }
    "stop" { Invoke-Stop }
    "selftest" { Invoke-SelfTest -Name $Command }
}
