"""
various convenience functions
"""

from puffbird.frame import FrameEngine


def puffy_to_long(table, *cols, **kwargs):
    """
    Transform the *"puffy"* table into a *long-format*
    :obj:`~pandas.DataFrame`.

    Parameters
    ----------
    table : :obj:`~pandas.DataFrame`
        A table with *"puffy"* columns.
    cols : str
        A selection of *"data columns"* to create the long dataframe with.
        If not given, the algorithm will use all *"data columns"*.
    iterable : callable or dict of callables, optional
        This function is called on each cell for each *"data column"*
        to create a new :obj:`~pandas.Series` object.
        If the *"data columns"* contains :obj:`dict`, :obj:`list`,
        :obj:`int`, :obj:`float`,
        :obj:`~numpy.array`, :obj:`~numpy.recarray`,
        :obj:`~pandas.DataFrame`, or :obj:`~pandas.Series` object types
        than the default iterable will handle these appropriately.
        When passing a dictionary of iterables, the keys should
        correspond to values in :obj:`~FrameEngine.datacols` (i.e.
        the *"data columns"* of the `table`). In this case, each column can
        have a custom iterable used. If a column's iterable is not
        specified the default iterable is used.
    max_depth : int or dict of ints, optional
        Maximum depth of expanding each cell, before the algorithm stops
        for each *"data column"*. If we set the max_depth to 3,
        for example,
        a *"data column"* consisting of 4-D :obj:`~numpy.array` objects
        will result in a :obj:`~pandas.DataFrame`
        where the *"data column"* cells contain
        1-D :obj:`~numpy.array` objects.
        If the arrays were 3-D, it will result in a
        long dataframe with scalars in each cell.
        Defaults to 3.
    dropna : bool, optional
        Drop rows in *long-format* :obj:`~pandas.DataFrame`,
        where **all** *"data columns"* are NaNs.
    cond : callable or dict of callables, optional
        This function should return `True` or `False` and accept a
        :obj:`~pandas.Series` object as an argument. If True, the algorithm
        will stop *"exploding"* a *"data column"*. The default `cond`
        argument suffices for all non-hashable types, such as
        :obj:`list` or :obj:`~numpy.array` objects. If you want
        to *"explode"* hashable types such as :obj:`tuple` objects, a
        custom `cond` callable has to be defined. However, it is
        recommended that hashable types are first converted into non-hashable
        types using a custom conversion function and the
        :obj:`~FrameEngine.col_apply` method.
    expand_cols : list-like, optional
        Specify a list of *"data columns"* to apply the
        :obj:`~FrameEngine.expand_col` method instead of *"exploding"*
        the column in the table.
        If all cells within a *"data column"* contains similarly
        constructed
        :obj:`~pandas.DataFrame` or :obj:`~pandas.Series` object types,
        the :obj:`~FrameEngine.expand_col` method can be used instead
        of *"exploding"* the *"data column"*. Default to None.
    shared_axes : dict, optional
        Specify if two or more *"data columns"* share axes
        (i.e. *"explosion"* iterations). The keyword
        will correspond to what the column will be called in the long
        dataframe. Each argument is a dictionary where the keys
        correspond to the names of the *"data columns"*, which share
        an axis, and the value correspond to the depth/axis is shared
        for each *"data column"*. `shared_axis` argument is usually defined
        for *"data columns"* that contain :obj:`~numpy.array` objects.
        For example, one *"data column"* may consists of one-dimensional
        timestamp arrays and another *"data column"* may consist of
        two-dimensional timeseries arrays where the first axis of the
        latter is shared with the zeroth axis of the former.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        A `long-format` :obj:`~pandas.DataFrame`.

    See Also
    --------
    FrameEngine.to_long
    FrameEngine.expand_col

    Notes
    -----
    If you find yourself writing custom `iterable` and `cond` arguments
    and believe these may be of general use, please open an
    `issue <https://github.com/gucky92/puffbird/issues>`_ or
    start a pull request.

    Examples
    --------
    >>> import pandas as pd
    >>> import puffbird as pb
    >>> df = pd.DataFrame({
    ...     'a': [[1,2,3], [4,5,6,7], [3,4,5]],
    ...     'b': [{'c':['asdf'], 'd':['ret']}, {'d':['r']}, {'c':['ff']}],
    ... })
    >>> df
                  a                              b
    0     [1, 2, 3]  {'c': ['asdf'], 'd': ['ret']}
    1  [4, 5, 6, 7]                   {'d': ['r']}
    2     [3, 4, 5]                  {'c': ['ff']}

    Now we can use the :obj:`puffy_to_long` function to create
    a `long-format` :obj:`~pandas.DataFrame`:

    >>> pb.puffy_to_long(df)
        index_level0  a_level0    a b_level0  b_level1     b
    0              0         0  1.0        c         0  asdf
    1              0         0  1.0        d         0   ret
    2              0         1  2.0        c         0  asdf
    3              0         1  2.0        d         0   ret
    4              0         2  3.0        c         0  asdf
    5              0         2  3.0        d         0   ret
    6              1         0  4.0        d         0     r
    7              1         1  5.0        d         0     r
    8              1         2  6.0        d         0     r
    9              1         3  7.0        d         0     r
    10             2         0  3.0        c         0    ff
    11             2         1  4.0        c         0    ff
    12             2         2  5.0        c         0    ff
    """

    return FrameEngine(table).to_long(*cols, **kwargs)
