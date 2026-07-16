param(
    [string]$Config = "configs/default.yaml",
    [string]$RunRoot = "runs"
)

$ErrorActionPreference = "Stop"
$Arms = @("baseline_matched", "multitask", "dbzd_full", "dbzd_stopgrad")
$Seeds = @(42, 43, 44)

foreach ($Arm in $Arms) {
    foreach ($Seed in $Seeds) {
        $FinalModel = Join-Path $RunRoot "${Arm}_s${Seed}/model_final.pt"
        if (Test-Path $FinalModel) {
            Write-Host "Skipping completed $Arm seed $Seed"
            continue
        }
        $Args = @(
            "train.py", "--config", $Config, "--run-root", $RunRoot,
            "--arm", $Arm, "--seed", "$Seed"
        )
        $Checkpoint = Join-Path $RunRoot "${Arm}_s${Seed}/checkpoint_latest.pt"
        if (Test-Path $Checkpoint) {
            $Args += "--resume"
        }
        python @Args
    }
}
