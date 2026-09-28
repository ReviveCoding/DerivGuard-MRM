"""Synthetic truth, observation, and fault-injection laboratories."""

from derivguard.synthetic.faults import (
    FAULT_DESCRIPTIONS,
    ValidationCase,
    clean_validation_case,
    inject_fault,
)
from derivguard.synthetic.generation import (
    SyntheticDesign,
    SyntheticTruth,
    UnsupportedTruthModelError,
    generate_truth,
    stratified_design,
    truth_capabilities,
)
from derivguard.synthetic.observation import ObservationConfig, ObservedSurface, observe_truth

__all__ = [
    "FAULT_DESCRIPTIONS",
    "ObservationConfig",
    "ObservedSurface",
    "SyntheticDesign",
    "SyntheticTruth",
    "UnsupportedTruthModelError",
    "ValidationCase",
    "clean_validation_case",
    "generate_truth",
    "inject_fault",
    "observe_truth",
    "stratified_design",
    "truth_capabilities",
]
