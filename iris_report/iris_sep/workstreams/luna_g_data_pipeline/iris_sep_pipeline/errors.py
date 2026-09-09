class PipelineError(ValueError):
    """Raised for an integrity violation; callers must fail closed."""


class DuplicateRecordError(PipelineError):
    """Raised when two records claim the same canonical identity."""


class ProtectedDataError(PipelineError):
    """Raised if a protected/non-synthetic record crosses the synthetic gate."""
