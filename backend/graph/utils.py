from typing import List, TypeVar

T = TypeVar("T")


def select_evenly(items: List[T], n: int) -> List[T]:
    """
    Select n items evenly spread across the list, instead of
    just taking the first n. Ensures later tasks in the plan
    still get covered even when the search budget is limited.
    """
    if n <= 0:
        return []

    if n >= len(items):
        return items

    step = len(items) / n

    indices = [int(i * step) for i in range(n)]

    return [items[i] for i in indices]