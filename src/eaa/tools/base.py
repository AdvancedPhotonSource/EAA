from typing import Any


class BaseTool:
    
    name: str = "base_tool"
    description: str = "A base tool for all tools."
    
    def __init__(self, *args, **kwargs):
        self.build()

    def build(self, *args, **kwargs):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
