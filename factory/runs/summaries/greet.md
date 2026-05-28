# Summary: greet

- Converged at iter 1 with min_satisfaction = 1.0
- Threshold = 0.9
- Judge notes: All three holdout scenarios fully pass. Missing --name exits with status 2, prints a stderr message mentioning '--name' (argparse's standard 'the following arguments are required: --name'), and produces no 'Hello, !' or traceback. Multi-word name 'Ada Lovelace' is preserved verbatim with the space intact and exits 0. --help exits 0 and the usage output contains both 'usage' and '--name'.

## Staged diff
```
 factory/.factory/config                            |  16 +
 factory/.factory/guardrail.sh                      | 155 +++++++
 factory/.factory/on_stop.sh                        |  26 ++
 factory/CLAUDE.md                                  |  49 +++
 factory/loop.sh                                    | 468 +++++++++++++++++++++
 factory/memory/decisions.md                        |  48 +++
 factory/memory/log.md                              |  47 +++
 factory/scenarios/greet_visible_01_basic.md        |   7 +
 factory/scenarios/greet_visible_02_file_exists.md  |   5 +
 .../holdout/greet_hidden_01_missing_name.md        |  12 +
 .../holdout/greet_hidden_02_multi_word_name.md     |  11 +
 .../scenarios/holdout/greet_hidden_03_help_flag.md |  11 +
 factory/tasks/greet.md                             |  25 ++
 greet.py                                           |  27 ++
 14 files changed, 907 insertions(+)
```

The loop did NOT commit. Human reviews and commits manually.
