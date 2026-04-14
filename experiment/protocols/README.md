# CV protocols

Place **speaker-disjoint** 5-fold split manifests here, for example `folds/fold_00.json` … `folds/fold_04.json`.

Each file should list train vs test utterance IDs (or paths) so that **no speaker appears in both** splits. CSS uses parallel speakers—always split by `speaker_id`, not by random files.

When folds are generated, remove the `.gitkeep` in `folds/` if you commit real JSON files (or keep it if folds stay local).
