# TODO long to puffy

from typing import (
    Dict, Union, Optional,
    Collection
)

from .frame import FrameEngine, DEFAULT_MAX_DEPTH


def puffy_to_long(
    table,
    datacols: Optional[Collection] = None,
    indexcols: Optional[Collection] = None,
    iterable: Union[callable, Dict[str, callable]] = iter,
    max_depth: Union[int, Dict[str, int]] = DEFAULT_MAX_DEPTH,
    dropna: bool = True,
    inplace: bool = False,
    handle_column_types: bool = True,
    enforce_identifier_string: bool = False,
    **shared_axes
):
    """
    Convenience function to convert puffy dataframe to a long format dataframe.
    """

    pf = FrameEngine(
        table, datacols=datacols, indexcols=indexcols, inplace=inplace,
        handle_column_types=handle_column_types,
        enforce_identifier_string=enforce_identifier_string
    )

    return pf.to_long(
        iterable=iterable, max_depth=max_depth,
        dropna=dropna, **shared_axes)
