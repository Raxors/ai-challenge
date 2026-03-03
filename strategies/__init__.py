from strategies.base import ContextStrategy
from strategies.sliding_window import SlidingWindowStrategy
from strategies.sticky_facts import StickyFactsStrategy
from strategies.branching import BranchingStrategy
from strategies.memory_layers import MemoryLayerStrategy

STRATEGY_REGISTRY = {}


def register_strategy(name, cls, requires_model=False):
    STRATEGY_REGISTRY[name] = {"cls": cls, "requires_model": requires_model}


def create_strategy(name, model=None, **kwargs) -> ContextStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Use: {list(STRATEGY_REGISTRY.keys())}")
    entry = STRATEGY_REGISTRY[name]
    if entry["requires_model"]:
        return entry["cls"](model=model, **kwargs)
    return entry["cls"](**kwargs)


def get_strategy_names() -> list[str]:
    return list(STRATEGY_REGISTRY.keys())


# Auto-register built-ins
register_strategy("sliding_window", SlidingWindowStrategy)
register_strategy("sticky_facts", StickyFactsStrategy, requires_model=True)
register_strategy("branching", BranchingStrategy)
register_strategy("memory_layers", MemoryLayerStrategy, requires_model=True)
