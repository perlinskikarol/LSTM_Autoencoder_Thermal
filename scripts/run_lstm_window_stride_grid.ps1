param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$dateTag = "20260605"
$datasetRoot = "Dane_przygotowane"
$runRoot = "runs\lstm_autoencoder"
$subjectId = "M1"
$comparisonDir = "runs\lstm_autoencoder\M1\window_stride_grid_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"

$excludedSessions = @(
    "M1_normalny_22.03.2026_12.23.25",
    "M1_normalny_31.03.2026_18.13.30",
    "M1_normalny_17.04.2026_13.46.42",
    "M1_prysznic_28.04.2026_13.17.36"
)

$valSessions = @(
    "M1_normalny_29.04.2026_18.11.21",
    "M1_normalny_04.05.2026_22.08.46",
    "M1_normalny_16.05.2026_14.12.39"
)

$testSessions = @(
    "M1_normalny_03.06.2026_22.49.00"
)

$configs = @(
    @{ Label = "1min_s5"; SeqLen = 60; Stride = 5; DatasetName = "lstm_autoencoder_1min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_1min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "1min_s10"; SeqLen = 60; Stride = 10; DatasetName = "lstm_autoencoder_1min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_1min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "3min_s5"; SeqLen = 180; Stride = 5; DatasetName = "lstm_autoencoder_3min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_3min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "3min_s10"; SeqLen = 180; Stride = 10; DatasetName = "lstm_autoencoder_3min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_3min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "5min_s5"; SeqLen = 300; Stride = 5; DatasetName = "lstm_autoencoder_5min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_5min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "5min_s10"; SeqLen = 300; Stride = 10; DatasetName = "lstm_autoencoder_5min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_5min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "10min_s5"; SeqLen = 600; Stride = 5; DatasetName = "lstm_autoencoder_10min_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_10min_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "10min_s10"; SeqLen = 600; Stride = 10; DatasetName = "lstm_autoencoder_10min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_10min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "12min_s5"; SeqLen = 720; Stride = 5; DatasetName = "lstm_autoencoder_12min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_12min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "12min_s10"; SeqLen = 720; Stride = 10; DatasetName = "lstm_autoencoder_12min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_12min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "15min_s5"; SeqLen = 900; Stride = 5; DatasetName = "lstm_autoencoder_15min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_15min_s5_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" },
    @{ Label = "15min_s10"; SeqLen = 900; Stride = 10; DatasetName = "lstm_autoencoder_15min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag"; RunName = "m1_15min_s10_fixednorm2_val3_test1_all_except_zero_shower_$dateTag" }
)

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Stage,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "=== $Stage ==="
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Stage failed ($LASTEXITCODE): $Stage"
    }
}

function Write-AnomalySessionSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RunDir
    )

    $base = Join-Path $RunDir "figures\session_scores"
    New-Item -ItemType Directory -Path $base -Force | Out-Null
    $scorePath = Join-Path $RunDir "scores_by_window.csv"
    $outputPath = Join-Path $base "all_anomaly_session_scores.csv"

    $rows = Import-Csv $scorePath |
        Where-Object { $_.session_state -ne "normalny" } |
        Group-Object session_key |
        ForEach-Object {
            $group = $_.Group
            $errors = $group | ForEach-Object { [double]($_.reconstruction_error -replace ",", ".") }
            $sorted = $errors | Sort-Object
            $p95Index = [Math]::Min(
                $sorted.Count - 1,
                [Math]::Max(0, [int][Math]::Floor(0.95 * ($sorted.Count - 1)))
            )
            $above = ($group | Where-Object { $_.is_anomaly -eq "true" }).Count
            [pscustomobject]@{
                session_key = $_.Name
                state = $group[0].session_state
                windows = $group.Count
                above_threshold = $above
                above_ratio = [Math]::Round($above / [Math]::Max($group.Count, 1), 8)
                mean_error = [Math]::Round(($errors | Measure-Object -Average).Average, 8)
                median_error = [Math]::Round($sorted[[Math]::Floor(($sorted.Count - 1) / 2)], 8)
                p95_error = [Math]::Round($sorted[$p95Index], 8)
                max_error = [Math]::Round(($errors | Measure-Object -Maximum).Maximum, 8)
            }
        }

    $rows | Sort-Object state, session_key |
        Export-Csv $outputPath -NoTypeInformation -Encoding UTF8
}

$runSpecs = @()

foreach ($config in $configs) {
    $datasetDir = Join-Path $datasetRoot $($config.DatasetName)
    $subjectDatasetDir = Join-Path $datasetDir $subjectId
    $runDir = Join-Path (Join-Path $runRoot $subjectId) $($config.RunName)
    $metricsPath = Join-Path $runDir "metrics.json"
    $reportPath = Join-Path $runDir "figures\REPORT.md"

    Write-Host ""
    Write-Host "##### $($config.Label): seq_len=$($config.SeqLen), stride=$($config.Stride) #####"

    if ($Force -or -not (Test-Path (Join-Path $subjectDatasetDir "dataset.npz"))) {
        Invoke-Checked "prepare $($config.Label)" {
            python -m src.prepare_lstm_dataset `
                --input-dir Pomiary `
                --output-dir $datasetDir `
                --seq-len $($config.SeqLen) `
                --stride $($config.Stride) `
                --skip-initial-sec 10 `
                --num-val-normal-sessions 3 `
                --num-test-normal-sessions 1 `
                --exclude-session-key $excludedSessions `
                --val-normal-session-key $valSessions `
                --test-normal-session-key $testSessions
        }
    } else {
        Write-Host "Dataset exists, skipping prepare: $subjectDatasetDir"
    }

    if ($Force -or -not (Test-Path $metricsPath)) {
        Invoke-Checked "train $($config.Label)" {
            python -m src.train_lstm_autoencoder `
                --dataset-dir $subjectDatasetDir `
                --output-dir $runRoot `
                --run-name $($config.RunName) `
                --epochs 200 `
                --batch-size 64 `
                --hidden-size 64 `
                --latent-size 32 `
                --num-layers 1 `
                --dropout 0 `
                --lr 0.001 `
                --weight-decay 0.00001 `
                --patience 10 `
                --threshold-quantile 0.99 `
                --seed 42 `
                --device auto
        }
    } else {
        Write-Host "Run exists, skipping train: $runDir"
    }

    if ($Force -or -not (Test-Path $reportPath)) {
        Invoke-Checked "report $($config.Label)" {
            python -m src.generate_lstm_visual_report --run-dir $runDir
        }
    } else {
        Write-Host "Report exists, skipping report: $reportPath"
    }

    Write-AnomalySessionSummary -RunDir $runDir
    $runSpecs += "$($config.Label)=$runDir"
}

$comparisonArgs = @()
foreach ($spec in $runSpecs) {
    $comparisonArgs += "--run"
    $comparisonArgs += $spec
}
$comparisonArgs += "--output-dir"
$comparisonArgs += $comparisonDir

Invoke-Checked "window/stride comparison" {
    python -m src.plot_lstm_window_length_comparison @comparisonArgs
}

Write-Host ""
Write-Host "Grid complete: $comparisonDir"
