from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class TokenLayout:
    grid: tuple[int, int]
    text_tokens: int

    def __post_init__(self):
        if (len(self.grid) != 2 or any(type(n) is not int or n <= 0 for n in self.grid)
                or type(self.text_tokens) is not int or self.text_tokens <= 0):
            raise ValueError("Expected positive H/W and text token count")

    @property
    def image_tokens(self):
        return math.prod(self.grid)

    @property
    def image_slice(self):
        return (0, self.image_tokens)

    @property
    def text_slice(self):
        return (self.image_tokens, self.total_tokens)

    @property
    def total_tokens(self):
        return self.image_tokens + self.text_tokens


def parse_block_scope(scope: str, block_count: int) -> frozenset[int]:
    if scope.strip() == "all":
        return frozenset(range(block_count))
    result = set()
    for part in scope.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if match is None:
            raise ValueError("block_scope must be all, indices, or inclusive ranges")
        start = int(match[1])
        end = int(match[2]) if match[2] else start
        if not 0 <= start <= end < block_count:
            raise ValueError("block_scope is outside native MOSS layers")
        result.update(range(start, end + 1))
    if not result:
        raise ValueError("block_scope cannot be empty")
    return frozenset(result)


@dataclass(frozen=True)
class AFMConfig:
    mode: str = "observe"
    strength: float = 0.05
    block_scope: str = "0-2"
    branch_mode: str = "positive_only"
    start_percent: float = 0.0
    end_percent: float = 1.0
    cutoff: float = 0.25
    soft_mask: bool = True
    mask_width: float = 0.05
    preserve_dc: bool = True
    head_chunk_size: int = 1
    query_chunk_size: int = 128
    max_workspace_mib: float = 512.0
    fail_mode: str = "raise"
    logging: str = "summary"

    def __post_init__(self):
        enums = {"mode": ("off", "observe", "edit"),
                 "branch_mode": ("positive_only", "negative_only", "both"),
                 "fail_mode": ("raise", "fallback_original"),
                 "logging": ("off", "summary")}
        for key, choices in enums.items():
            if getattr(self, key) not in choices:
                raise ValueError(f"Invalid {key}")
        for key in ("strength", "start_percent", "end_percent", "cutoff",
                    "mask_width", "max_workspace_mib"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not 0 <= self.start_percent <= self.end_percent <= 1:
            raise ValueError("Expected 0 <= start_percent <= end_percent <= 1")
        if not 0 < self.cutoff < 1 or not 0 < self.mask_width <= 1:
            raise ValueError("Invalid cutoff or mask_width")
        if self.max_workspace_mib <= 0:
            raise ValueError("max_workspace_mib must be positive")
        for key in ("head_chunk_size", "query_chunk_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("preserve_dc", "soft_mask"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")

    @property
    def bypass(self):
        return self.mode == "off" or (self.mode == "edit" and self.strength == 0)
