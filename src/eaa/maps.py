import enum


class ExtendedStrEnum(enum.StrEnum):
    
    @classmethod
    def contains(cls, value: str) -> bool:
        return value in list(cls)


class OpenAIModels(ExtendedStrEnum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_O1_MINI = "o1-mini"
    GPT_O1_PREVIEW = "o1-preview"
    GPT_O1 = "o1"
    GPT_O3_MINI = "o3-mini"
    GPT_O3_MINI_HIGH = "o3-mini-high"
    GPT_O3_PREVIEW = "o3-preview"
    GPT_O3 = "o3"
    GPT_O4_MINI = "o4-mini"
    GPT_O4_PREVIEW = "o4-preview"
    GPT_O4 = "o4"
    
    
class AnthropicModels(ExtendedStrEnum):
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
