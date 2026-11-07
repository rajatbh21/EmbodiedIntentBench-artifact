"""The task registry.

``contract.task_id`` resolves a task here, so the runner never imports a task module and
``task_id`` never becomes a branch in the core. Adding object detection later means adding a
package under ``tasks/`` and one line in this file.
"""

from __future__ import annotations

from missionbench.registry import Registry
from missionbench.schemas.task_spec import TaskSpec
from missionbench.tasks.base import TaskDefinition
from missionbench.tasks.human_search_segmentation.ground_truth import TASK_ID
from missionbench.tasks.human_search_segmentation.task import HumanSearchSegmentationTask

__all__ = ["resolve_task", "task_registry"]

task_registry: Registry[TaskDefinition] = Registry("task")

task_registry.register(TASK_ID, HumanSearchSegmentationTask)


def resolve_task(task_spec: TaskSpec) -> TaskDefinition:
    """Return the task implementation for a loaded specification.

    Keyed on the specification's own ``task_id``, so a fixture cannot be paired with an
    implementation that does not match it.
    """
    return task_registry.create(task_spec.task_id, task_spec=task_spec)
