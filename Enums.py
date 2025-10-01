from enum import Flag, auto

class SortDirection(Flag):
    """Enumeration for sorting direction."""
    ROW_LEFT_TO_RIGHT = auto()
    ROW_RIGHT_TO_LEFT = auto()
    COLUMN_TOP_TO_BOTTOM = auto()
    COLUMN_BOTTOM_TO_TOP = auto()

class WhatToSort(Flag):
    """Enumeration for what to sort."""
    BACKGROUND = auto()
    FOREGROUND = auto()
    ALL = auto()
    