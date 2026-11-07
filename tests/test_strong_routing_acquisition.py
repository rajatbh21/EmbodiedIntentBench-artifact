import math

import pytest

from experiments.lake_tracking import camera_ray
from experiments.strong_routing_20261030.acquisition import (
    geometry_candidate, project_point, segment_intersects_box,
)
from experiments.wide_area_missions.runtime import Trajectory


def test_projection_inverts_existing_camera_rays():
    pose = {'position_m': [3., -7., 25.], 'quaternion_xyzw': [0., 0., 0., 1.]}
    for pixel in ((100., 100.), (479.5, 359.5), (800., 650.)):
        origin, ray = camera_ray(pixel, (720, 960), pose, 1.2)
        assert project_point(origin + ray * 30, pose['position_m']) == pytest.approx(pixel)


def test_fresh_pass_has_repeated_geometry_and_fixed_clock():
    targets = {'a': [0., -5., 0.], 'b': [0., 0., 0.], 'c': [0., 5., 0.]}
    result = geometry_candidate(targets, 18., lambda x, y: 0., [])
    assert result['passed']
    assert all(n >= 4 for n in result['geometry_visible_capture_counts'].values())
    trajectory = Trajectory(result['trajectory'])
    assert trajectory.duration_s == 42.
    assert math.dist(trajectory.sample(0)[0], trajectory.sample(36)[0]) == pytest.approx(36.)
    assert trajectory.sample(18)[1] == pytest.approx((0., 1.5, 0.))
    assert trajectory.sample(38)[1] == (0., 0., 0.)


def test_obstacle_and_insufficient_height_fail_geometry_gate():
    targets = {'a': [0., -5., 0.], 'b': [0., 0., 0.], 'c': [0., 5., 0.]}
    assert not geometry_candidate(targets, 10., lambda x, y: 0., [])['passed']
    wall = [('wall', [-2., -100., 0.], [-1., 100., 100.])]
    result = geometry_candidate(targets, 18., lambda x, y: 0., wall)
    assert not result['passed']
    assert all(n == 0 for n in result['geometry_visible_capture_counts'].values())


def test_finite_ray_box_intersection_and_parallel_axis():
    assert segment_intersects_box([0, 0, 0], [2, 0, 0], [.5, -1, -1], [1, 1, 1])
    assert not segment_intersects_box([0, 3, 0], [2, 3, 0], [.5, -1, -1], [1, 1, 1])
    assert not segment_intersects_box([0, 0, 0], [.2, 0, 0], [.5, -1, -1], [1, 1, 1])
