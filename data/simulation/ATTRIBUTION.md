# Simulator asset attribution

The simulator assets are separate from the benchmark's code license.

| Material | Upstream creator/source | License and changes |
|---|---|---|
| `derived/*.dae` | OpenRobotics, [Harmonic Terrain Objects v1](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Harmonic%20Terrain%20Objects/1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); isolated tree, bush and rock triangles translated to local origins |
| `overrides/terrain_model.sdf` | OpenRobotics, [Harmonic Terrain v1](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Harmonic%20Terrain/1) | CC BY 4.0; collision heightmap extent aligned with unchanged visual extent |
| Downloaded terrain, terrain objects, house and water | OpenRobotics; versioned sources recorded in `assets.lock.json` | CC BY 4.0; original downloaded geometry/textures retained except the declared terrain SDF override |
| Downloaded standing-person mesh/textures | Marina Kollmitz, hosted by OpenRobotics; [Standing person v3](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing%20person/3) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/); unchanged mesh and textures |

The benchmark authors specify placements, camera settings and the fixed trajectory
in `scenario.json`. The world builder omits demo-specific control plugins and the
upstream background-tree visual, adds the derived props at authored positions,
and supplies conservative collision proxies. All original upstream credits are
retained. No endorsement by the original asset creators is implied.
