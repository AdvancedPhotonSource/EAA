class MaxRoundsReached(Exception):
    
    def __init__(self, message: str = "Maximum number of rounds reached."):
        self.message = message
        super().__init__(self.message)
