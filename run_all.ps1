param(
    [string]$Config = "configs/default.yaml",
    [string]$RunRoot = "runs"
)

$ErrorActionPreference = "Stop"
$Arms = @("baseline_matched", "multitask", "dbzd_full", "dbzd_stopgrad")
$Seeds = @(42, 43, 44)
$ExpectedRevision = python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['experiment_revision'])" $Config

foreach ($Arm in $Arms) {
    foreach ($Seed in $Seeds) {
        $RunDir = Join-Path $RunRoot "${Arm}_s${Seed}"
        $FinalModel = Join-Path $RunDir "model_final.pt"
        $ResolvedConfig = Join-Path $RunDir "resolved_config.yaml"
        $CurrentRevision = ""
        if (Test-Path $ResolvedConfig) {
            $CurrentRevision = python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8')).get('experiment_revision',''))" $ResolvedConfig
        }
        if ((Test-Path $FinalModel) -and $CurrentRevision -eq $ExpectedRevision) {
            Write-Host "Skipping completed $Arm seed $Seed"
            continue
        }
        $Args = @(
            "train.py", "--config", $Config, "--run-root", $RunRoot,
            "--arm", $Arm, "--seed", "$Seed"
        )
        $Checkpoint = Join-Path $RunDir "checkpoint_latest.pt"
        if ((Test-Path $Checkpoint) -and $CurrentRevision -eq $ExpectedRevision) {
            $Args += "--resume"
        } elseif ($CurrentRevision -and $CurrentRevision -ne $ExpectedRevision) {
            Write-Host "Re-running stale revision $CurrentRevision for $Arm seed $Seed"
        }
        python @Args
    }
}
