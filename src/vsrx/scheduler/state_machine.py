from __future__ import annotations

from vsrx.domain.enums import JobState

_ALLOWED: dict[JobState, set[JobState]] = {
    JobState.DISCOVERED: {JobState.PROBED, JobState.CANCELLED, JobState.FAILED},
    JobState.PROBED: {JobState.ANALYZED, JobState.ENCODING, JobState.CANCELLED, JobState.FAILED},
    JobState.ANALYZED: {JobState.MASKED, JobState.CANCELLED, JobState.FAILED},
    JobState.MASKED: {JobState.RECONSTRUCTING, JobState.CANCELLED, JobState.FAILED},
    JobState.RECONSTRUCTING: {
        JobState.INPAINTING,
        JobState.QUALITY_CHECK,
        JobState.CANCELLED,
        JobState.FAILED,
    },
    JobState.INPAINTING: {
        JobState.QUALITY_CHECK,
        JobState.RETRY_PENDING,
        JobState.CANCELLED,
        JobState.FAILED,
    },
    JobState.QUALITY_CHECK: {
        JobState.RECONSTRUCTING,
        JobState.INPAINTING,
        JobState.RETRY_PENDING,
        JobState.ENCODING,
        JobState.REVIEW_REQUIRED,
        JobState.CANCELLED,
        JobState.FAILED,
    },
    JobState.RETRY_PENDING: {
        JobState.RECONSTRUCTING,
        JobState.INPAINTING,
        JobState.REVIEW_REQUIRED,
        JobState.FAILED,
        JobState.CANCELLED,
    },
    JobState.ENCODING: {
        JobState.DONE,
        JobState.REVIEW_REQUIRED,
        JobState.FAILED,
        JobState.CANCELLED,
    },
    JobState.REVIEW_REQUIRED: {
        JobState.ENCODING,
        JobState.DONE,
        JobState.CANCELLED,
        JobState.FAILED,
    },
    JobState.DONE: set(),
    JobState.FAILED: {JobState.RETRY_PENDING, JobState.CANCELLED},
    JobState.CANCELLED: {JobState.RETRY_PENDING},
}


def can_transition(current: JobState, target: JobState) -> bool:
    return current == target or target in _ALLOWED.get(current, set())


def assert_transition(current: JobState, target: JobState) -> None:
    if not can_transition(current, target):
        raise ValueError(f"invalid job-state transition: {current.value} -> {target.value}")
