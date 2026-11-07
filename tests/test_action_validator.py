"""Policy action validation, privacy enforcement, and switch counting.

The policy is the subject under evaluation, so it is treated as untrusted input: every
malformed or illegal action must be recorded and replaced, never raised.
"""

from __future__ import annotations

import pytest

from missionbench.schemas.configuration import (
    ConfigCatalog,
    Configuration,
    Placement,
    Precision,
    Strategy,
)
from missionbench.schemas.contract import PrivacyLevel
from missionbench.simulator.action_validator import (
    TRANSMITTED_PAYLOAD_PARAMETER,
    ActionOutcome,
    ActionValidator,
    TransmittedPayload,
    is_switch,
    privacy_permits,
)

ALLOWED = ("CFG_LOCAL_LIGHT", "CFG_LOCAL_STRONG", "CFG_REMOTE_STRONG")


@pytest.fixture
def validator(catalog) -> ActionValidator:
    return ActionValidator(
        catalog=catalog,
        allowed_config_ids=ALLOWED,
        privacy_level=PrivacyLevel.REMOTE_ALLOWED,
    )


# --- valid actions --------------------------------------------------------------------


@pytest.mark.parametrize("config_id", ALLOWED)
def test_an_allowed_configuration_is_accepted(validator: ActionValidator, config_id: str) -> None:
    action = validator.validate(config_id, current_config_id=None)
    assert action.is_valid
    assert action.outcome is ActionOutcome.VALID
    assert action.config_id == config_id
    assert not action.was_substituted


# --- invalid actions ------------------------------------------------------------------


def test_an_unknown_configuration_is_rejected(validator: ActionValidator) -> None:
    action = validator.validate("CFG_DOES_NOT_EXIST", current_config_id=None)
    assert action.outcome is ActionOutcome.UNKNOWN_CONFIG
    assert action.config_id == "CFG_LOCAL_LIGHT"


def test_a_configuration_outside_the_episode_pool_is_rejected(catalog) -> None:
    validator = ActionValidator(
        catalog=catalog,
        allowed_config_ids=("CFG_LOCAL_LIGHT",),
        privacy_level=PrivacyLevel.REMOTE_ALLOWED,
    )
    action = validator.validate("CFG_REMOTE_STRONG", current_config_id=None)
    assert action.outcome is ActionOutcome.NOT_ALLOWED
    assert "allowed pool" in action.reason


@pytest.mark.parametrize("junk", [None, 42, 3.5, ["CFG_LOCAL_LIGHT"], object()])
def test_a_non_string_action_does_not_crash_the_episode(
    validator: ActionValidator, junk: object
) -> None:
    action = validator.validate(junk, current_config_id=None)
    assert action.outcome is ActionOutcome.MALFORMED
    assert action.config_id == "CFG_LOCAL_LIGHT"


def test_the_requested_action_is_preserved_for_the_record(validator: ActionValidator) -> None:
    action = validator.validate("CFG_NOPE", current_config_id="CFG_LOCAL_STRONG")
    assert action.requested == "CFG_NOPE"
    assert action.to_dict()["requested_config_id"] == "CFG_NOPE"
    assert action.to_dict()["config_id"] == "CFG_LOCAL_STRONG"
    assert action.to_dict()["outcome"] == "unknown_config"


def test_a_non_string_request_serialises_without_pretending_to_be_an_id(
    validator: ActionValidator,
) -> None:
    payload = validator.validate(42, current_config_id=None).to_dict()
    assert payload["requested_config_id"] is None
    assert payload["requested_repr"] == "42"


# --- substitution ---------------------------------------------------------------------


def test_an_invalid_action_keeps_the_current_configuration(validator: ActionValidator) -> None:
    action = validator.validate("CFG_NOPE", current_config_id="CFG_REMOTE_STRONG")
    assert action.config_id == "CFG_REMOTE_STRONG"
    assert "kept current configuration" in action.reason


def test_the_first_invalid_action_falls_back_to_the_safe_default(
    validator: ActionValidator,
) -> None:
    action = validator.validate("CFG_NOPE", current_config_id=None)
    assert action.config_id == "CFG_LOCAL_LIGHT"
    assert "safe fallback" in action.reason


def test_an_unacceptable_current_configuration_is_not_kept(catalog) -> None:
    """An episode's declared initial_config_id never passed validation, so it is re-checked."""
    validator = ActionValidator(
        catalog=catalog,
        allowed_config_ids=("CFG_LOCAL_LIGHT",),
        privacy_level=PrivacyLevel.REMOTE_ALLOWED,
    )
    action = validator.validate("CFG_NOPE", current_config_id="CFG_REMOTE_STRONG")
    assert action.config_id == "CFG_LOCAL_LIGHT"
    assert "safe fallback" in action.reason


def test_the_fallback_is_configurable(catalog) -> None:
    validator = ActionValidator(
        catalog=catalog,
        allowed_config_ids=ALLOWED,
        privacy_level=PrivacyLevel.REMOTE_ALLOWED,
        fallback_config_id="CFG_LOCAL_STRONG",
    )
    assert validator.validate("CFG_NOPE", current_config_id=None).config_id == "CFG_LOCAL_STRONG"


@pytest.mark.parametrize(
    ("fallback", "allowed", "privacy"),
    [
        ("CFG_NOT_REAL", ALLOWED, PrivacyLevel.REMOTE_ALLOWED),
        ("CFG_REMOTE_STRONG", ("CFG_LOCAL_LIGHT",), PrivacyLevel.REMOTE_ALLOWED),
        ("CFG_REMOTE_STRONG", ALLOWED, PrivacyLevel.LOCAL_ONLY),
    ],
)
def test_an_unusable_fallback_fails_at_construction(
    catalog, fallback: str, allowed: tuple[str, ...], privacy: PrivacyLevel
) -> None:
    """Discovering mid-episode that there is no legal action would leave the runner stuck."""
    with pytest.raises(ValueError, match="unusable"):
        ActionValidator(
            catalog=catalog,
            allowed_config_ids=allowed,
            privacy_level=privacy,
            fallback_config_id=fallback,
        )


# --- privacy ---------------------------------------------------------------------------


def test_local_only_blocks_remote_execution(catalog) -> None:
    validator = ActionValidator(
        catalog=catalog,
        allowed_config_ids=ALLOWED,
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )
    action = validator.validate("CFG_REMOTE_STRONG", current_config_id="CFG_LOCAL_LIGHT")

    assert action.outcome is ActionOutcome.PRIVACY_VIOLATION
    assert action.config_id == "CFG_LOCAL_LIGHT", "the violating configuration must not run"
    assert "privacy level 'local_only'" in action.reason


def test_local_only_still_permits_local_configurations(catalog) -> None:
    validator = ActionValidator(
        catalog=catalog,
        allowed_config_ids=ALLOWED,
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )
    assert validator.validate("CFG_LOCAL_STRONG", current_config_id=None).is_valid


def _remote(payload: str | None) -> Configuration:
    parameters = {} if payload is None else {TRANSMITTED_PAYLOAD_PARAMETER: payload}
    return Configuration(
        config_id="CFG_R",
        model_id="M",
        strategy=Strategy(
            placement=Placement.REMOTE, precision=Precision.FP16, parameters=parameters
        ),
    )


def _local() -> Configuration:
    return Configuration(
        config_id="CFG_L",
        model_id="M",
        strategy=Strategy(placement=Placement.LOCAL, precision=Precision.INT8),
    )


@pytest.mark.parametrize(
    ("privacy", "configuration", "permitted"),
    [
        (PrivacyLevel.REMOTE_ALLOWED, _remote(None), True),
        (PrivacyLevel.REMOTE_ALLOWED, _local(), True),
        (PrivacyLevel.LOCAL_ONLY, _local(), True),
        (PrivacyLevel.LOCAL_ONLY, _remote(TransmittedPayload.FEATURES.value), False),
        (PrivacyLevel.FEATURES_ONLY, _local(), True),
        (PrivacyLevel.FEATURES_ONLY, _remote(TransmittedPayload.FEATURES.value), True),
        (PrivacyLevel.FEATURES_ONLY, _remote(TransmittedPayload.RAW_INPUT.value), False),
    ],
)
def test_privacy_rules(
    privacy: PrivacyLevel, configuration: Configuration, permitted: bool
) -> None:
    assert privacy_permits(privacy, configuration) is permitted


def test_features_only_fails_closed_on_an_undeclared_payload() -> None:
    """Forgetting the parameter must not quietly grant permission."""
    assert not privacy_permits(PrivacyLevel.FEATURES_ONLY, _remote(None))


def test_a_features_configuration_is_selectable_under_features_only() -> None:
    catalog = ConfigCatalog([_local(), _remote(TransmittedPayload.FEATURES.value)])
    validator = ActionValidator(
        catalog=catalog,
        allowed_config_ids=("CFG_L", "CFG_R"),
        privacy_level=PrivacyLevel.FEATURES_ONLY,
        fallback_config_id="CFG_L",
    )
    assert validator.validate("CFG_R", current_config_id=None).is_valid


# --- switch counting ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        (None, "CFG_LOCAL_LIGHT", False),  # nothing to switch away from
        ("CFG_LOCAL_LIGHT", "CFG_LOCAL_LIGHT", False),  # re-selection is not a switch
        ("CFG_LOCAL_LIGHT", "CFG_REMOTE_STRONG", True),
    ],
)
def test_switch_counting(previous: str | None, current: str, expected: bool) -> None:
    assert is_switch(previous, current) is expected


def test_a_substituted_action_does_not_count_as_a_switch(validator: ActionValidator) -> None:
    """The count follows what actually ran, not what the policy failed to ask for."""
    action = validator.validate("CFG_NOPE", current_config_id="CFG_LOCAL_STRONG")
    assert not is_switch("CFG_LOCAL_STRONG", action.config_id)


def test_switch_count_over_a_selection_sequence(validator: ActionValidator) -> None:
    selections = [
        "CFG_LOCAL_LIGHT",
        "CFG_LOCAL_LIGHT",
        "CFG_REMOTE_STRONG",
        "CFG_NOPE",  # invalid: keeps CFG_REMOTE_STRONG, not a switch
        "CFG_LOCAL_LIGHT",
    ]
    current: str | None = None
    switches = 0
    for requested in selections:
        action = validator.validate(requested, current_config_id=current)
        switches += is_switch(current, action.config_id)
        current = action.config_id

    assert switches == 2
    assert current == "CFG_LOCAL_LIGHT"
