Set-Location $PSScriptRoot\..\..\apps\api

$candidatePythons = @()

if ($env:ANIME_TRACK_PYTHON) {
  $candidatePythons += $env:ANIME_TRACK_PYTHON
}

if ($env:CONDA_PREFIX) {
  $candidatePythons += (Join-Path $env:CONDA_PREFIX "python.exe")
}

$candidatePythons += @(
  "%USERPROFILE%\.conda\envs\Anime_Track\python.exe",
  "C:\ProgramData\anaconda3\envs\Anime_Track\python.exe"
)

$pythonExe = $candidatePythons |
  Where-Object { $_ -and (Test-Path $_) } |
  Select-Object -First 1

if (-not $pythonExe) {
  $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCommand) {
    $pythonExe = $pythonCommand.Source
  }
}

if (-not $pythonExe) {
  throw "No usable Python interpreter found. Please select the Anime_Track interpreter in PyCharm or set ANIME_TRACK_PYTHON."
}

Write-Host "Using Python: $pythonExe"
& $pythonExe -m uvicorn main:app --reload
