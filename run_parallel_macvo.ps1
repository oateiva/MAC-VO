param(
    [string]$ExpDir = "Config/Experiment/MACVO",
    [string]$SeqDir = "Config/Sequence/TartanAir_Sample_Dataset",
    [int]$MaxParallel = 2,
    [string]$CondaEnv = "AITraining12",
    [int]$niters = 10
)

# -------------------------
# 1) Activate conda env
# -------------------------
conda activate $CondaEnv
Write-Host "Activated conda environment: $CondaEnv"
Write-Host ""

# -------------------------
# 2) Ensure script runs from project root
# -------------------------
Set-Location -Path $PSScriptRoot

$jobs = @()

# Collect experiment and sequence YAMLs
$expFiles = Get-ChildItem -Path $ExpDir -Filter *.yaml
$seqFiles = Get-ChildItem -Path $SeqDir -Filter *.yaml

if ($expFiles.Count -eq 0) {
    Write-Host "No experiment YAMLs found in $ExpDir"
    exit 1
}
if ($seqFiles.Count -eq 0) {
    Write-Host "No sequence YAMLs found in $SeqDir"
    exit 1
}

Write-Host "Found $($expFiles.Count) experiment YAML(s) and $($seqFiles.Count) sequence YAML(s)."
Write-Host "Iterations: $niters"
Write-Host "Max parallel jobs: $MaxParallel"
Write-Host ""

# ------------------------------------------
# 3) Precompute all jobs to know total count
# ------------------------------------------
$jobConfigs = @()
for ($i = 1; $i -le $niters; $i++) {
    foreach ($exp in $expFiles) {
        foreach ($seq in $seqFiles) {
            $jobConfigs += [PSCustomObject]@{
                Run = $i
                Exp = $exp
                Seq = $seq
            }
        }
    }
}

$totalJobs = $jobConfigs.Count
Write-Host "Total jobs to run: $totalJobs"
Write-Host ""

# -------------------------
# 4) Submit jobs (with limit)
# -------------------------
$submitted = 0

foreach ($cfg in $jobConfigs) {

    # Throttle: wait until fewer than MaxParallel are running
    while ((Get-Job -State Running).Count -ge $MaxParallel) {
        $runningJobs   = Get-Job -State Running
        $completedJobs = Get-Job -State Completed

        $percent = if ($totalJobs -gt 0) {
            [math]::Round(($completedJobs.Count / $totalJobs) * 100, 1)
        } else {
            0
        }

        $runningNames = if ($runningJobs) {
            ($runningJobs.Name -join ", ")
        } else {
            "none"
        }

        Write-Progress -Activity "Running MACVO jobs" `
                       -Status "Completed: $($completedJobs.Count)/$totalJobs | Running: $runningNames" `
                       -PercentComplete $percent

        Start-Sleep -Seconds 1
    }

    # Create a descriptive name for the job (used only in progress display)
    $jobName = "Run$($cfg.Run)_Odom_$($cfg.Exp.BaseName)_Seq_$($cfg.Seq.BaseName)"
    $submitted++
    $logFile = ".\logs\$jobName.log"

    # ensure log directory exists
    if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

    $jobs += Start-Job -Name $jobName -ScriptBlock {
        param($rootDir, $odomYaml, $dataYaml, $runIndex, $logFile)

        Set-Location -Path $rootDir

        # Run python and redirect ALL output to the log
        python macvo.py --odom $odomYaml --data $dataYaml *> $logFile

    } -ArgumentList $PSScriptRoot, $cfg.Exp.FullName, $cfg.Seq.FullName, $cfg.Run, $logFile
}

# -------------------------
# 5) Final wait loop with progress only
# -------------------------
while ((Get-Job -State Running).Count -gt 0) {
    $runningJobs   = Get-Job -State Running
    $completedJobs = Get-Job -State Completed

    $percent = if ($totalJobs -gt 0) {
        [math]::Round(($completedJobs.Count / $totalJobs) * 100, 1)
    } else {
        0
    }

    $runningNames = if ($runningJobs) {
        ($runningJobs.Name -join ", ")
    } else {
        "none"
    }

    Write-Progress -Activity "Running MACVO jobs" `
                   -Status "Completed: $($completedJobs.Count)/$totalJobs | Running: $runningNames" `
                   -PercentComplete $percent

    Start-Sleep -Seconds 1
}

Write-Progress -Activity "Running MACVO jobs" -Completed

Write-Host "Collecting job outputs:`n"
Wait-Job -Job $jobs | Out-Null
Receive-Job -Job $jobs

Write-Host "`nAll MACVO runs finished."
