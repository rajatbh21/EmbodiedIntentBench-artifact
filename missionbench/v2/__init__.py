"""EmbodiedIntentBench legacy visual layer: a minimal visual closed loop.

The original core used abstract mission fixtures without imagery. This legacy
visual layer adds a raster world between mission state and perception execution.
It is retained for compatibility and shared interfaces; it is not the entrypoint
for the paper's Gazebo-based experiments. Use ``tools/artifact.py`` for those
experiments and consult ``docs/ARTIFACT.md`` for their reproduction scope.

    2D aerial world (GeoTIFF / image)
      -> predefined UAV trajectory (position from mission time)
      -> position-dependent camera crop (windowed raster read)
      -> configuration-selection policy (the V1 Policy protocol, unchanged)
      -> image-based executor (prediction depends on RGB content)
      -> mission latency moves the UAV; busy executors skip observations
      -> capture-time ground truth scores the prediction
      -> mission-level evaluation (V1 empirical metric semantics)

Dependency boundary
-------------------
The V1 core stays runtime-dependency-free. Everything under this subpackage may use
``numpy``/``Pillow`` (and ``rasterio`` for GeoTIFF worlds), installed via the optional
``missionbench[v2]`` extra. Later replay experiments reuse some of these interfaces
and install their lightweight image dependencies through ``missionbench[replay]``.

Honesty
-------
V2 is low fidelity by design. Latency and energy are *simulated* (configured per
executor), the targets are *synthetic* rescue markers composited over real aerial
imagery, and results must not be described as realistic UAV perception performance.
V2 validates architecture, interfaces, closed-loop semantics, and mission-level
trade-offs -- not flight physics, hardware, or real-world model quality.
"""
