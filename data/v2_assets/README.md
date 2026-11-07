# V2.2 target assets

This directory holds the **manifest** for image-based human target assets — and, by
design, **no actual human imagery**. The repository ships no licensed person asset:
committing one requires verified provenance (source, creator, license, redistribution
permission), and none has been supplied yet. Until an owner places a licensed asset
here and records it in a manifest, every image-asset scenario and the observability
experiment fail with an actionable error. That blocked state is intentional and honest.

## What an owner must supply

1. An RGBA PNG person cutout (alpha = mask), **or** an RGB image plus a separate
   binary mask image.
2. A manifest entry (start from `manifest_template.json`, rename to `manifest.json`)
   with **every** provenance field filled truthfully:
   - `source.provider`, `source.license`, `source.redistribution_allowed` are required;
   - `checksum_sha256` must match the file (`shasum -a 256 <file>`);
   - `physical.nominal_width_m` / `nominal_height_m` — real-world size, never inferred;
   - `view_type` — `conventional` (a normal photo; usable only as a controlled
     model-integration diagnostic), `aerial` (a genuinely overhead capture), or
     `procedural` (a generated test silhouette). **Never** relabel a conventional photo
     as aerial.

## Hard rules (from the benchmark's asset policy)

- No images scraped from search engines; no unknown-license files; no invented
  provenance; no real person's image without a lawful project-owned basis.
- Assets whose `redistribution_allowed` is false stay local and uncommitted
  (`.gitignore` covers common image extensions here).
- Procedural silhouettes may drive automated tests and pipeline smoke runs, but are
  never presented as realistic person targets or as evidence of model performance.

## Commands

```bash
python -m missionbench.v2.cli validate-assets --manifest data/v2_assets/manifest.json
python -m missionbench.v2.cli inspect-assets  --manifest data/v2_assets/manifest.json \
  --output results/v2_assets
python -m missionbench.v2.cli run-observability \
  --config data/v2_scenarios/observability_img1.json --output results/v2_observability
```
