# Evaluation figure data and alternative visualizations

The supplied numeric sources support the evaluations in Figs. 4–6 and §5.4 of
the paper. The plotting scripts render alternative visualizations: their artwork,
panel layouts, and some displayed precision differ from the final manuscript.
The optional `fig7` summarizes §5.4 and is not a figure in the manuscript.
Rendering needs Python and Matplotlib; it does not run neural models or a
simulator. The original `figure_data.json` is preserved byte-for-byte.

| Figure | Experiment | Verification source |
|---|---|---|
| [4](fig4.pdf) | Depot full-bank recall, fixed-policy mission success and frozen predicted recall | Original [numeric case/frame rows](sources/depot_rows.json), cross-checked against the [archived ledger excerpt](sources/depot_summary.json) |
| [5](fig5.pdf) | Lake urgency, connectivity and battery comparisons | Released validation reference runs and urgency comparison |
| [6](fig6.pdf) | Causal/no-skip latency sweep, policy ranking and a predeclared timeline | Released validation metrics, paired outcomes and timeline |
| [7](fig7.pdf) (optional diagnostic) | Model-pool feasibility and evidence accumulation (§5.4) | Released validation condition classes, summary and evidence funnel |

## Verify the data and regenerate the figures

From the repository root:

```bash
python examples/paper_figures/verify_sources.py
python -m pip install -e '.[report]'
python examples/paper_figures/make_figures.py
```

The first command uses only the Python standard library. It checks source hashes,
all figure values, denominators and cross-table consistency. The second script
writes vector `fig4.pdf` through `fig7.pdf`, matching 300 dpi PNGs, and
`render_receipt.json`. The checked-in [verification receipt](verification.json)
records the checks and verifier hash.

To keep generated files outside the source directory:

```bash
python examples/paper_figures/verify_sources.py --output results/paper-figure-check.json
python examples/paper_figures/make_figures.py --output results/paper-figures
```

Both scripts work when invoked by an absolute path from another directory.
Times New Roman is preferred, then Liberation Serif, then Matplotlib's bundled
DejaVu Serif. The receipt records the actual font name and file checksum, without
a machine-specific path. Output hashes depend on the installed Matplotlib and
font versions. The same software and fonts produce deterministic files; numerical
verification is independent of those rendering dependencies.

To use an existing, locally licensed Times New Roman installation that Matplotlib
does not discover automatically, add `--font-dir PATH_TO_LOCAL_FONT_DIRECTORY`.
Font files are not copied or redistributed in this repository. The checked-in
figures use Times New Roman at 20 pt.

## What is and is not reproduced

**Figure 4 is reproduced from original numeric outputs.** The package contains
144 original fixed-policy case rows (nine configurations on 16 conditions) and
all 2,160 frame/configuration rows (nine configurations on 240 captures). The
verifier recomputes mission successes, observed recall and target-observation-
weighted predicted recall, then cross-checks the archived evidence-ledger excerpt.
These source files were extracted only after both original full-file hashes
matched the hashes preserved in `figure_data.json`. The extraction retains all
recall rows and all fixed-policy case fields except dispatch schedules and
redundant selection counts. The optional extraction utility documents this step:

```bash
python examples/paper_figures/extract_depot_sources.py --source PATH_TO_ORIGINAL_DEPOT_OUTPUTS
```

The raw Depot recordings, mask prediction bank and complete per-run receipts are
not included. Verification cannot rerun neural inference or all Depot missions,
nor independently rehash the absent original full files. The bundled hashes
verify the supplied numeric extracts and excerpt, not the absent full ledger.

**The alternative plots for Figures 5–6 and the optional diagnostic reuse the
validation tables.** The package reads
[../validation/](../validation/) directly, so it does not duplicate `runs.csv`.
Its verification recomputes the plotted reductions from these tables; it does not
execute policies. The [validation guide](../../docs/VALIDATION.md) explains how to
rerun the experiments from the supplied replay inputs. To verify a newly
regenerated set of validation tables, pass its directory explicitly:

```bash
python examples/paper_figures/verify_sources.py --validation PATH_TO_VALIDATION_TABLES
```

The required CSVs and timeline must match the frozen scientific sources. The
validation summary's code-hash metadata may change with packaging, but its
feasibility and evidence-funnel values must still agree. Energy cross-checks
against the independently rounded urgency table retain the original 0.005 J
tolerance; stored figure data and primary table reductions are unchanged.

The four figures describe different experiments and denominators. Figure 4 uses
16 Depot conditions; Figure 5 uses 48 Lake comparison conditions, or 24 per factor
level; Figures 6-7 use all 72 Lake conditions. The Figure 6 timeline is the
predeclared fixed-local-Mask example, which fails under both execution protocols.
Figure 7 includes evaluator-only hindsight schedules, not deployable routers.

## Caption examples

[captions.tex](captions.tex) contains captions and descriptions for these
alternative plots, including the optional diagnostic. They are not the current
manuscript captions. Each plot is 16 inches wide with 20 pt text; at a 7-inch
two-column print width the text becomes 8.75 pt.

Only portability and metadata handling changed in the drawing script. No new
data, policies, thresholds or scientific figure values were introduced.
