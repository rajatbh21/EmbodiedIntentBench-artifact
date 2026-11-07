# Verification

The receipts below record checks on the source revisions identified in each
receipt. They do not imply that every check was repeated after later
documentation or packaging changes. Replay uses stored predictions; it is not
new neural inference or a new simulator flight.

## Reference replay and validation

A complete check in a fresh Linux/Python 3.12.14 environment on 4 November
2026 produced the following results. The environment and dependency versions
are recorded in the [environment receipt](verification_environment.json) and
[requirements](requirements.txt).

| Check | Result | Evidence |
| --- | --- | --- |
| Input and source provenance | Passed | [Input receipt](verification_input_validation.json) |
| Isolated smoke replay | 27/27 exact results | [Smoke receipt](verification_smoke_validation.json) |
| Reference replay | 648/648 exact scientific results | [Replay receipt](verification_replay_validation.json) |
| Validation | 1,944 contract reruns, 5,184 causal/no-skip runs, and 80 hindsight searches over 13,332 candidates; all 25 successful schedules independently reproduced | [Validation receipt](verification_validation.json) |
| Sensitivity and tables | 144 settings; 23 CSVs and two JSON reports reproduced exactly, apart from the declared figure-source digest | [Validation receipt](verification_validation.json) |

For aggregate reports, the first comparison found ten last-bit mean-energy
differences due to Python 3.12's changed floating-point `sum` (maximum
1.14e-13 J). CSVs and timelines already matched. The reporting wrapper uses
the original sequential addition; checks on [Python 3.10](verification_python310_aggregation.json)
and [Python 3.12](verification_python312_aggregation.json) confirm the resulting
aggregation. Reports were regenerated from the completed execution receipts;
the experiments were not rerun for this reporting adjustment. The
[source mapping](release_source_changes.json) records the presentation-only
figure-renderer digest change.

## Subsequent checks

The [5 November update receipt](paper_alignment_validation.json) records 830
passing tests, 17 skipped historical-artifact tests, a 27/27 exact smoke replay,
input and source checks, and 268 figure source-value checks. It also identifies
the files and scope of that check. The figure scripts produce alternative
visualizations based on the same data, rather than exact reproductions of the
final manuscript artwork; see the [figure guide](../examples/paper_figures/README.md).

Earlier checks include a [648-mission replay](replay_validation.json), an
[isolated 27-mission replay](relocation_validation.json),
[fresh inference on nine masks](native_validation.json), and a
[42-second Gazebo acquisition](simulation_validation.json). The inference and
Gazebo checks used existing host dependencies and do not establish a fresh
installation of the complete simulator and ML stack.

The receipts retain the original paths and hashes recorded during earlier verification runs.
