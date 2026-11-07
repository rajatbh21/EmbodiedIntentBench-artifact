# Fresh simulator recording

The release includes a portable scenario for one **42-second inspection** on the
textured lake map. It preserves the authored trajectory, camera, people, terrain
and scenery placement of a recording used in the routing study. Gazebo simulates
the multicopter's rigid-body motion and rotor thrust; a fixed controller follows
the trajectory. This command creates a new recording and leaves the published
replay inputs unchanged.

| Item | Value |
|---|---|
| Simulator | Gazebo Harmonic / Sim 8, headless Ogre2 rendering |
| Terrain extent | 1,200 × 1,200 m |
| Selected path | One smooth 36 m inspection pass; 36 s movement + 6 s settling |
| Camera | 960 × 720 RGB and evaluator-only object labels |
| Stored captures | 21 at 2 s intervals; first 18 are motion captures |
| Controller | Fixed trajectory with velocity feed-forward and ideal odometry |
| Physical checks | Actual tracking error, terrain clearance, conservative obstacle bounds |

This is a local inspection on a large map. It is not a complete eight-sector
mission, a field flight, or a new routing result. New captures may differ in pixels
across machines or graphics drivers.

## Install the optional simulator

Install [Gazebo Harmonic using the official Ubuntu instructions](https://gazebosim.org/docs/harmonic/install_ubuntu/).
Use a Linux machine with headless OpenGL/EGL rendering and local Gazebo transport
available. The worker uses the system Python that provides `gz.transport13` and
`gz.msgs10`, plus NumPy and Pillow. The parent command uses the project's Python
environment. These may be different interpreters.

```bash
python -m pip install -e '.[replay]'
python tools/simulate.py doctor --python /usr/bin/python3
```

The dependency probe does not establish that rendering or local IPC works. The
capture command checks those paths. Containers must permit the simulator's local
transport and render device access. No physical drone is controlled.

## Prepare and verify assets

```bash
python tools/simulate.py prepare
python tools/simulate.py verify
```

Preparation downloads four version-pinned public scenery archives (about 371 MB
compressed) and the standing-person asset (about 25 MB of required files). Allow
roughly 1 GB of disk space. Every archive and required file is verified with the
checksums in [assets.lock.json](../data/simulation/assets.lock.json). Downloads are
stored under the ignored `.cache/simulation/` directory. Subsequent captures do
not use the network. `prepare --offline` only succeeds when all required inputs
are already present or supplied through its optional local-source arguments.

The lock includes Harmonic Terrain v1, Harmonic Terrain Objects v1, Lake House v3,
Coast Waves 2 v3, and Standing person v3. Upstream attribution is retained in
[the asset credits](../data/replay/asset_attribution.json). The lake scenery is
CC BY 4.0; Standing person is CC0 and credits Marina Kollmitz. The release also
ships small derived tree, bush and rock meshes under CC BY 4.0. These isolate
triangles from the upstream terrain-object mesh and place them at fixed authored
locations. One SDF override aligns the terrain collision height to its unchanged
visual height of 201.66 m. The simulator builder removes the upstream background
tree visual, retains the selected original scenery, and uses fixed conservative
collision proxies for the added props. These adaptations match the stored
scenario and do not use model outputs.

## Capture and inspect

```bash
python tools/simulate.py capture --output results/simulation_demo
```

Choose a new output directory for every run. Existing recordings are never
overwritten. Successful capture writes:

- `preview.png`: an RGB image from the middle of the inspection.
- `recording.json` and 21 `frame_*.npz` files: synchronized images and body poses.
- `physical_route_validation.json`: tracking, clearance and obstacle checks.
- `simulation_summary.json`: concise completion and image checks.
- `world.sdf`, `worker.log`, and `gazebo.log`: the generated scene and diagnostics.

Outputs contain runtime-local paths for auditability and remain ignored by Git.
The shipped scenario and assets contain no workstation paths. RGB images are
public routing inputs; semantic labels, target identities and desired trajectory
states are evaluator/controller data and must not be supplied to a routing policy.
Use [the replay instructions](REPRODUCING.md) to compare routers against the
existing fixed inputs. Fresh acquisition alone does not infer new model outputs
or recompute the benchmark's routing results.
