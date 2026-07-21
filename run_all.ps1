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
        $Summary = Join-Path $RunDir "summary.json"
        $CurrentRevision = ""
        $CompletedSummary = $false
        if (Test-Path $ResolvedConfig) {
            $CurrentRevision = python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8')).get('experiment_revision',''))" $ResolvedConfig
        } elseif (Test-Path $Summary) {
            $CurrentRevision = python -c "import json,sys; print((json.load(open(sys.argv[1], encoding='utf-8')).get('config') or {}).get('experiment_revision',''))" $Summary
        }
        if (Test-Path $Summary) {
            $CompletedSummary = (python -c "import json,sys; print(str(bool(json.load(open(sys.argv[1], encoding='utf-8')).get('completed'))).lower())" $Summary) -eq "true"
        }
        if ($CurrentRevision -and $CurrentRevision -ne $ExpectedRevision) {
            throw "Refusing to overwrite stale revision $CurrentRevision in $RunDir"
        }
        if (((Test-Path $FinalModel) -or $CompletedSummary) -and $CurrentRevision -eq $ExpectedRevision) {
            Write-Host "Skipping completed $Arm seed $Seed"
            if ($Arm -eq "dbzd_full") {
                python scripts/validate_alpha.py --run-dir $RunDir
            }
            continue
        }
        $Args = @(
            "train.py", "--config", $Config, "--run-root", $RunRoot,
            "--arm", $Arm, "--seed", "$Seed"
        )
        $Checkpoint = Join-Path $RunDir "checkpoint_latest.pt"
        if ((Test-Path $Checkpoint) -and $CurrentRevision -eq $ExpectedRevision) {
            $Args += "--resume"
        }
        python @Args
        if ($Arm -eq "dbzd_full") {
            python scripts/validate_alpha.py --run-dir $RunDir
        }
    }
}
