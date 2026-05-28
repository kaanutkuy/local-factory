# Summary: slugify

- Converged at iter 1 with min_satisfaction = 1.0
- Threshold = 0.9
- Judge notes: All slugify edge scenarios pass: 'foo_bar baz' → 'foo_bar-baz' (underscore preserved as literal _, space → -), 'Café Résumé' → 'café-résumé' (Unicode letters preserved and lowercased, no ASCII folding), 'Hello,  World!' → 'hello-world' (comma+double-space collapse to single -, trailing ! dropped, no leading/trailing hyphens). All three greet regression scenarios also still pass: basic and multi-word greet print verbatim with exit 0, missing --name exits 2 with stderr mentioning 'name' and no traceback, --help exits 0 with 'usage' and '--name' in output. Every scenario fully satisfies its 'satisfied when' criteria.

## Staged diff
```
 factory/.factory/config                    |  2 +-
 factory/loop.sh                            | 17 +++++++++++-
 factory/memory/log.md                      | 16 ++++++++++++
 factory/runs/summaries/slugify.md          | 19 ++++++++++++++
 factory/scenarios/holdout/slugify_edges.md | 42 ++++++++++++++++++++++++++++++
 factory/scenarios/slugify_basic.md         | 17 ++++++++++++
 factory/tasks/slugify.md                   | 24 +++++++++++++++++
 slugify.py                                 | 17 ++++++++++++
 8 files changed, 152 insertions(+), 2 deletions(-)
```

The loop did NOT commit. Human reviews and commits manually.
