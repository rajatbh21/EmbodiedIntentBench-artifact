"""The attribution control fixes source/placement even during infeasibility."""
import copy

from experiments.strong_routing_20261030 import remote_control as remote, study


def test_fixed_remote_control_does_not_silently_fallback():
    class Delegate:
        def select(self, **inputs):
            return dict(action_id='local', candidates=[dict(action_id=remote.ACTION, legal=True, feasible=False)])
    selected = remote.RemoteYolo(Delegate()).select()
    assert selected['action_id'] == remote.ACTION
    assert selected['reason'] == 'fixed_control_ignores_resource_forecast'


def test_privacy_still_blocks_fixed_remote():
    class Delegate:
        def select(self, **inputs):
            return dict(action_id='local', candidates=[dict(action_id=remote.ACTION, legal=False, feasible=False)])
    assert remote.RemoteYolo(Delegate()).select()['action_id'] is None


def test_supplement_changes_only_policy_and_run_identity():
    parent = dict(grid=study.make_grid({f'site_{i}':dict(duration_s=42) for i in range(6)}))
    before = copy.deepcopy(parent)
    grid = remote.make_grid(parent)
    assert len(grid) == len({r['run_id'] for r in grid}) == 72
    assert parent == before
    original = [r for r in parent['grid'] if r['policy'] == 'fixed_local_yolo']
    for a, b in zip(original, grid):
        assert {k for k in a if a[k] != b[k]} == {'policy','run_id'}
        assert b['policy'] == remote.NAME
