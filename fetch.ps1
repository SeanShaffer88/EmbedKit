# Download the pinned fixtures into fixtures/ and verify SHA-256:
# the ONNX Runtime Windows package, plus the e2e model + vocab.
# Nothing fetched here is ever committed; fixtures/ is gitignored, and
# ONNX Runtime is a runtime component the DLL loads dynamically — the
# fetch exists for the e2e proof and as a known-good source for users.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pins = @{}
Get-Content pins.txt | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
    $k, $v = $_ -split '=', 2; $pins[$k.Trim()] = $v.Trim()
}
New-Item -ItemType Directory -Force fixtures | Out-Null

function Fetch([string]$Url, [string]$File, [string]$Want) {
    $path = Join-Path fixtures $File
    if (-not (Test-Path $path)) {
        Write-Host "downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile $path
    }
    $got = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $Want) {
        throw "SHA-256 MISMATCH for $File`n  pinned: $Want`n  actual: $got"
    }
    Write-Host "verified $File ($got)"
}

Fetch $pins['ORT_WIN_URL'] $pins['ORT_WIN_FILE'] $pins['ORT_WIN_SHA256']
Fetch $pins['TEST_MODEL_URL'] $pins['TEST_MODEL_NAME'] $pins['TEST_MODEL_SHA256']
Fetch $pins['TEST_VOCAB_URL'] $pins['TEST_VOCAB_NAME'] $pins['TEST_VOCAB_SHA256']

# Extract the runtime package into fixtures/ort (stable layout for the
# e2e test: fixtures/ort/lib/onnxruntime.dll on Windows). The zip holds
# a single top-level folder; Expand-Archive handles .zip natively.
Remove-Item -Recurse -Force fixtures/ort, fixtures/tmp-ort -ErrorAction SilentlyContinue
Expand-Archive -Path (Join-Path fixtures $pins['ORT_WIN_FILE']) -DestinationPath fixtures/tmp-ort
$inner = Get-ChildItem fixtures/tmp-ort -Directory | Select-Object -First 1
Move-Item $inner.FullName fixtures/ort
Remove-Item -Recurse -Force fixtures/tmp-ort
Write-Host "runtime ready under fixtures/ort; model at fixtures/$($pins['TEST_MODEL_NAME']); vocab at fixtures/$($pins['TEST_VOCAB_NAME'])"
Write-Host "e2e: `$env:EK_ORT_LIB='fixtures/ort/lib/onnxruntime.dll'; `$env:EK_MODEL='fixtures/$($pins['TEST_MODEL_NAME'])'; `$env:EK_VOCAB='fixtures/$($pins['TEST_VOCAB_NAME'])'; cargo test --test e2e -- --ignored --nocapture"
