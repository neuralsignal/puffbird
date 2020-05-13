"""
Various utility functions
"""


from pandas.api.types import is_hashable


class SeriesIsHashable:
    """Test if `pandas.Series` is hashable
    """

    def __call__(self, series):
        return series.dtype != object or is_hashable(tuple(series))

    def __repr__(self):
        return 'series_is_hashable'


series_is_hashable = SeriesIsHashable()
