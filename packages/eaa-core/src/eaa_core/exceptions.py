class MaxRoundsReached(Exception):
    """Raised when a feedback loop exceeds its round budget."""

    def __init__(self, message: str = "Maximum number of rounds reached."):
        """Initialize the exception."""
        self.message = message
        super().__init__(self.message)
