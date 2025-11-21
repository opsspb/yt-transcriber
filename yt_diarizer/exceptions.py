"""Custom exceptions for the yt_diarizer package."""


class DependencyError(RuntimeError):
    """Raised when a required external dependency is missing or broken."""


class PipelineError(RuntimeError):
    """Raised when any step of the pipeline fails."""
