"""
FrameEngine class
-----------------
Class to transform a wide pandas DataFrame, as it may be obtained from a
database model like datajoint, into a long dataframe format optimal for
plotting and groupby computations.
"""

import re
from typing import (
    Dict, Union, Optional,
    Collection, Sequence
)
from numbers import Number

import numpy as np
import pandas as pd

from puffbird.err import PuffbirdError
from puffbird.utils import series_is_hashable
from puffbird.callables import CallableContainer

RESERVED_COLUMNS = {
    "apply_result", "max_depth", "dropna", "iterable",
    "datacols", "indexcols", "handle_column_types",
    "enforce_identifier_string", "aggfunc"
}
DEFAULT_MAX_DEPTH = 3
DEFAULT_AGGFUNC = CallableContainer(list)
DEFAULT_CONDITION = series_is_hashable
DATACOL_REGEX = "{datacol}(_level)?[1-9]*$"
# iterable handles dicts and pandas series correctly
DEFAULT_ITERABLE = CallableContainer(iter)
DEFAULT_ITERABLE.add(lambda x: iter(x), (np.ndarray, set))
DEFAULT_ITERABLE.add(lambda x: x, (dict, pd.Series, Number, list, tuple))
DEFAULT_ITERABLE.add(
    # stacks all columns
    lambda x: x.stack(level=list(range(x.columns.nlevels))),
    pd.DataFrame
)
DEFAULT_ITERABLE.add(
    lambda x: pd.DataFrame(x).stack(level=list(range(x.columns.nlevels))),
    np.recarray
)


class FrameEngine:
    """
    Class to handle and transform a :obj:`pandas.DataFrame` object.

    Parameters
    ----------
    table : :obj:`~pandas.DataFrame`
        A table with singular :obj:`~pandas.Index` columns, where each
        column corresponds to a specific data type. :obj:`~pandas.MultiIndex`
        columns will be made singular with the
        :obj:`~pandas.MultiIndex.to_flat_index`
        method. It is recommended that all columns and index names are
        identifier string types. Individual cells within `datacols` columns
        may have arbitrary objects in them, but cells within `indexcols`
        columns must be hashable.
    datacols : list-like, optional
        The columns in `table` that are considered **"data"**. For
        example, columns where each cell is a :obj:`numpy.array` object.
        If None, all columns are considered `datacols` columns,
        unless `indexcols` is specified. Defaults to None.
    indexcols : list-like, optional
        The columns in `table` that are immutable or hashable types,
        e.g. strings or integers. These may correspond to *"metadata"*  that
        describe or specify the `datacols` columns. If None,
        only the index of the `table`, which may be :obj:`~pandas.MultiIndex`,
        are considered `indexcols` columns. If `datacols` is specified and
        `indexcols` is None, then the remaining columns are also added to the
        index of `table`. Defaults to None.
    inplace : bool, optional
        If possible do not copy the `table` object. Defaults to False.
    handle_column_types : bool, optional
        If True, try to convert string column types into identifier strings.
        Integer and tuple column types will always be converted to
        string column types. Defaults to True.
    enforce_identifier_string : bool, optional
        If True, check if all columns are identifier string types and throw
        an error if this is not the case. Defaults to True.

    Notes
    -----
    A `table` has singular :obj:`~pandas.Index` columns, where each column
    corresponds to a specific data type. These types of tables are often
    fetched from databases that use data models such as
    `datajoint <https://datajoint.io>`_.
    The `table` often needs to be transformed, so that various
    computations such as :obj:`~pandas.DataFrame.groupby` can be performed or
    the data can be plotted easily with packages such as
    `seaborn <https://seaborn.pydata.org>`_.
    In the `table`, the columns and the index names are considered
    together and divided into `datacols` and `indexcols`. *"Data columns"*  are
    usually columns that contain Python objects that are iterable and need
    to be *"exploded"*  in order to convert these columns into numeric or other
    immutable data types. This is why I call these types of tables *"puffy"*
    dataframes.
    *"Index columns"*  usually contain other information,
    often considered *"metadata"*, that uniquely identify each row. Each row
    for a specific column is considered to have the same data type and can
    thus be *"exploded"*  the same way. Missing data (NaNs) are allowed.

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
    >>> engine = pb.FrameEngine(df)

    The :obj:`FrameEngine` instance has various methods that allow for
    quick manipulation of this *"puffy"*  dataframe. For example,
    we can create a long dataframe using the :func:`~FrameEngine.to_long`
    method:

    >>> engine.to_long()
        index_col_0 b_level0  b_level1     b  a_level0    a
    0             0        c         0  asdf         0  1.0
    1             0        c         0  asdf         1  2.0
    2             0        c         0  asdf         2  3.0
    3             0        d         0   ret         0  1.0
    4             0        d         0   ret         1  2.0
    5             0        d         0   ret         2  3.0
    6             1        d         0     r         0  4.0
    7             1        d         0     r         1  5.0
    8             1        d         0     r         2  6.0
    9             1        d         0     r         3  7.0
    10            2        c         0    ff         0  3.0
    11            2        c         0    ff         1  4.0
    12            2        c         0    ff         2  5.0
    """

    def __init__(
        self,
        table,
        datacols: Optional[Collection] = None,
        indexcols: Optional[Collection] = None,
        inplace: bool = False,
        handle_column_types: bool = True,
        enforce_identifier_string: bool = True
    ):
        if isinstance(table, pd.Series):
            table = _process_table_when_series(table)
        elif not isinstance(table, pd.DataFrame):
            table = _process_table_when_unknown_object(table)

        truth = RESERVED_COLUMNS & set(table.columns)
        if truth:
            raise PuffbirdError(f"Dataframe table has columns "
                                f"that are reserved: {truth}")

        if not inplace:
            table = table.copy()

        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.to_flat_index()

        table, datacols, indexcols = _process_column_types(
            table, datacols, indexcols
        )

        # table index must be a multiindex
        if not isinstance(table.index, pd.MultiIndex):
            table.index = pd.MultiIndex.from_frame(table.index.to_frame())

        table = _enforce_identifier_column_types(
            table, handle_column_types, enforce_identifier_string
        )

        # check table index and column types
        _check_table_column_types(table, enforce_identifier_string)

        # check if index is unique
        if not table.index.is_unique:
            raise PuffbirdError("Each row for all index columns "
                                "must be a unique set.")

        # assign table
        self._table = table

    @property
    def table(self):
        """
        :obj:`~pandas.DataFrame` passed during initialization.
        """
        return self._table

    @property
    def datacols(self):
        """
        Tuple of the *"data columns"* in the `table`.

        See Also
        --------
        pandas.DataFrame.columns
        """
        return tuple(self._table.columns)

    @property
    def indexcols(self):
        """
        Tuple of the *"index columns"* in the `table`.

        See Also
        --------
        pandas.MultiIndex.names
        """
        return tuple(self._table.index.names)

    def __repr__(self):
        return self.table.__repr__()

    def __str__(self):
        return self.table.__str__()

    def to_long(
        self, *,
        iterable: Union[callable, Dict[str, callable]] = DEFAULT_ITERABLE,
        max_depth: Union[int, Dict[str, int]] = DEFAULT_MAX_DEPTH,
        dropna: bool = True,
        reindex: bool = False,
        cond: Union[int, Dict[str, int]] = DEFAULT_CONDITION,
        expand_cols: Optional[Sequence[str]] = None,
        **shared_axes: dict
    ) -> pd.DataFrame:
        """
        Transform the *"puffy"* table into a *long-format*
        :obj:`~pandas.DataFrame`.

        Parameters
        ----------
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
        FrameEngine.cols_to_long
        puffy_to_long

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
        >>> engine = pb.FrameEngine(df)

        Now we can use the :obj:`~FrameEngine.to_long` method to create
        a `long-format` :obj:`~pandas.DataFrame`:

        >>> engine.to_long()
            index_col_0 b_level0  b_level1     b  a_level0    a
        0             0        c         0  asdf         0  1.0
        1             0        c         0  asdf         1  2.0
        2             0        c         0  asdf         2  3.0
        3             0        d         0   ret         0  1.0
        4             0        d         0   ret         1  2.0
        5             0        d         0   ret         2  3.0
        6             1        d         0     r         0  4.0
        7             1        d         0     r         1  5.0
        8             1        d         0     r         2  6.0
        9             1        d         0     r         3  7.0
        10            2        c         0    ff         0  3.0
        11            2        c         0    ff         1  4.0
        12            2        c         0    ff         2  5.0
        """

        expand_cols = [] if expand_cols is None else expand_cols
        truth = set(expand_cols) - set(self.datacols)
        if truth:
            raise PuffbirdError(f"Keys '{truth}' in 'expand_cols' not "
                                f"in 'data columns' {self.datacols}")

        # check shared axes arguments for correct formatting
        _check_shared_axes_argument(shared_axes, self.datacols, self.indexcols)

        # convert max_depth correctly
        max_depth, default_max_depth = _mapping_variable_converter(
            self.table, max_depth, DEFAULT_MAX_DEPTH, "max_depth")
        # iterable dictionary
        iterable, default_iterable = _mapping_variable_converter(
            self.table, iterable, DEFAULT_ITERABLE, "iterable")
        # get condition
        cond, default_cond = _mapping_variable_converter(
            self.table, cond, DEFAULT_CONDITION, "cond")

        # iterate of each data column
        for m, (datacol, series) in enumerate(self.table.items()):
            if datacol in expand_cols:
                # this only works if all objects within a series are dataframes
                names = set(series.index.names)
                _df = self.expand_col(
                    datacol, reset_index=False, dropna=dropna
                )
                # handle multiindex
                if isinstance(_df.columns, pd.MultiIndex):
                    _df.columns = _df.columns.to_flat_index()
                else:
                    # this way it is known where the column came from
                    _df.rename(
                        columns=lambda x: f"{datacol}_{x}", inplace=True
                    )
                _df.reset_index(inplace=True)
            else:
                if dropna:
                    series = series.dropna()
                # set first depth
                n = 0
                # if series already not object skip
                # TODO different conditions meet
                while (
                    not cond.get(datacol, default_cond)(series)
                    and max_depth.get(datacol, default_max_depth) > n
                ):
                    # superstack pandas.Series object
                    series = self._superstack_series(
                        series, datacol,
                        iterable.get(datacol, default_iterable), dropna,
                        _get_col_name(datacol, n, shared_axes)
                    )
                    n += 1

                # convert series to frame
                names = set(series.index.names)
                _df = series.reset_index()

            # merge with previous dataframe
            if not m:
                df = _df
            else:
                on = list(names & set(df.columns))
                df = pd.merge(
                    df, _df, on=on, how="outer", suffixes=("", f"_{datacol}")
                )

        # reindex if necessary:
        if reindex:
            index = list(set(df.columns) - set(self.datacols))
            df.set_index(index, inplace=True)

        return df

    @staticmethod
    def _superstack_series(series, datacol, iterable, dropna, col_name):
        # apply series iterable (iter is default for sequences)
        # series.index is already assumed to be multi index
        # transform into dataframe
        # this should automatically infer types
        table = series.apply(
            lambda x: pd.Series(iterable(x)),
            convert_dtype=True
        )
        # rename columns and stack dataframe
        if isinstance(table.columns, pd.MultiIndex):
            # rename columns properly
            # give columns index a name
            column_names = []
            for index, name in table.columns.names:
                if name is None:
                    name = f"{col_name}_{index}"
                else:
                    name = f"{col_name}_{name}"
                column_names.append(name)
            # rename column names
            table.rename_axis(
                columns=column_names,
                inplace=True
            )
            # stack all columns
            levels = list(range(table.columns.nlevels))
            series = table.stack(levels=levels, dropna=dropna)
        else:
            # give columns index a name
            table.columns.name = (
                col_name
                if table.columns.name is None
                else f"{col_name}_{table.columns.name}"
            )
            # stack table to single series
            series = table.stack(dropna=dropna)
        # give series a datacol name
        series.name = datacol
        return series

    def __getitem__(self, key):
        # if it is a dataframe return new instance of FrameEngine
        # TODO indexing indexcolumns
        selected_table = self.table[key]

        if isinstance(selected_table, pd.DataFrame):
            # creates a new instance
            return type(self)(selected_table)
        else:
            return selected_table

    def cols_to_long(self, *cols: str, **kwargs) -> pd.DataFrame:
        """
        Transform the *"puffy"* table into a *long-format*
        :obj:`~pandas.DataFrame`.

        Parameters
        ----------
        cols : str
            A list of *"data columns"* to create the long dataframe with.
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
        puffy_to_long

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
        >>> engine = pb.FrameEngine(df)

        Now we can use the :obj:`~FrameEngine.cols_to_long` method to create
        a `long-format` :obj:`~pandas.DataFrame`:

        >>> engine.cols_to_long('a', 'b')
            index_col_0 b_level0  b_level1     b  a_level0    a
        0             0        c         0  asdf         0  1.0
        1             0        c         0  asdf         1  2.0
        2             0        c         0  asdf         2  3.0
        3             0        d         0   ret         0  1.0
        4             0        d         0   ret         1  2.0
        5             0        d         0   ret         2  3.0
        6             1        d         0     r         0  4.0
        7             1        d         0     r         1  5.0
        8             1        d         0     r         2  6.0
        9             1        d         0     r         3  7.0
        10            2        c         0    ff         0  3.0
        11            2        c         0    ff         1  4.0
        12            2        c         0    ff         2  5.0
        """
        return self[list(cols)].tolong(**kwargs)

    def expand_col(
        self,
        col: str,
        reset_index: bool = True,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Expand a column that contain
        :obj:`~pandas.DataFrame` or :obj:`~pandas.Series` object types
        to create a single `long-format` :obj:`~pandas.DataFrame`.

        Parameters
        ----------
        col : str
            The *"data column"* to expand.
        reset_index : bool, optional
            Whether to reset the index of the new `long-format`
            :obj:`~pandas.DataFrame`. Defaults to True.
        dropna : bool, optional
            Whether to drop NaNs in the *"data column"*. If False and
            NaNs exist, this will currently result in an error.
            Defaults to True.

        Returns
        -------
        :obj:`~pandas.DataFrame` or :obj:`~pandas.Series`
            A `long-format` :obj:`~pandas.DataFrame` or :obj:`~pandas.Series`.

        See Also
        --------
        FrameEngine.to_long
        FrameEngine.cols_to_long
        puffy_to_long
        """

        if dropna:
            series = self.table[col].dropna()
        else:
            # produces an error if nan's are in the series
            series = self.table[col]

        long_df = pd.concat(
            list(series), keys=series.index,
            names=series.index.names,
            sort=False
        )

        if reset_index:
            return long_df.reset_index()
        else:
            return long_df

    def col_apply(
        self,
        func: callable,
        col: str,
        new_col_name: Optional[str] = None,
        assign_to_index: Optional[bool] = None,
        **kwargs
    ):
        """
        Apply a function to a specific column in each row in the `table`.

        Parameters
        ----------
        func : callable
            Function to apply. The function cannot return a
            :obj:`~pandas.Series` object.
        col : str
            Name of *"data column"*.
        new_col_name : str, optional
            Name of computed new column. If None, this will be set
            to the name of the column; i.e. the name of the column will be
            overwritten. Defaults to None.
        assign_to_index : bool, optional
            Assign new column as *"index column"*,
            instead of as *"data column"*..
        kwargs : dict
            Keyword Arguments passed each function call.

        Returns
        -------
        self
        """
        if new_col_name is None:
            new_col_name = col
        if assign_to_index is None:
            assign_to_index = col in self.indexcols
        # apply function
        series = _select_frame(self.table, col).apply(func, **kwargs)
        if isinstance(series, pd.DataFrame):
            raise PuffbirdError("The function 'func' cannot return "
                                "a `pandas.Series` object.")
        # assign output
        self._assign_output_series(series, new_col_name, assign_to_index)
        return self

    def apply(
        self,
        func: callable,
        new_col_name: Optional[callable],
        *args: str,
        assign_to_index: bool = False,
        map_kws: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Apply a function to each row in the `table`.

        Parameters
        ----------
        func : callable
            Function to apply. The function cannot return a
            :obj:`~pandas.Series` object.
        new_col_name : str
            Name of computed new col. If None, `new_col_name` will be
            "apply_result".
        args : tuple
            Arguments passed to function. Each argument should be an
            *"index column"* or *"data column"* in the `table`.
            Thus, the argument will correspond to the cell value for each row.
        assign_to_index : bool, optional
            Assign new column as *"index column"*,
            instead of as *"data column"*..
        map_kws : dict
            Same as args just as keyword arguments.
        kwargs: dict
            Keyword arguments passed to function as is.

        Returns
        -------
        self
        """
        map_kws = {} if map_kws is None else map_kws

        if new_col_name is None:
            new_col_name = "apply_result"
        # apply function
        series = self.table.reset_index().apply(
            lambda x: func(
                *(x[arg] for arg in args),
                **{key: x[arg] for key, arg in map_kws.items()},
                **kwargs
            ),
            axis=1, result_type="reduce"
        )
        if isinstance(series, pd.DataFrame):
            raise PuffbirdError("The function 'func' cannot return "
                                "a `pandas.Series` object.")
        # assign output
        self._assign_output_series(series, new_col_name, assign_to_index)
        return self

    def _assign_output_series(self, series, new_col_name, assign_to_index):
        """
        assign a series to a particular column or index name
        """
        if assign_to_index:
            if new_col_name in self.indexcols:
                index = self.table.index.to_frame(False)
                index[new_col_name] = series
                self.table.index = pd.MultiIndex.from_frame(index)
            else:
                self.table[new_col_name] = series
                self.table.set_index(
                    [new_col_name],
                    drop=True,
                    append=True,
                    inplace=True,
                    verify_integrity=True
                )
        else:
            if new_col_name in self.indexcols:
                raise PuffbirdError(f"Column name '{new_col_name}' already "
                                    "assigned to index columns; cannot "
                                    "assign to data columns. Choose "
                                    "different name.")
            self.table[new_col_name] = series

    def drop(
        self,
        *columns: str,
        skip: bool = False,
        skip_index: bool = False,
        skip_data: bool = False
    ):
        """
        Drop columns in place.

        Parameters
        ----------
        columns : str
        skip : bool
        skip_index : bool
        skip_data : bool

        Returns
        -------
        self

        See Also
        --------
        pandas.DataFrame.drop
        """
        not_found = set(columns) - (set(self.datacols) | set(self.indexcols))
        if not_found and not skip:
            raise PuffbirdError(f"Columns '{not_found}' are not in "
                                "'data columns' or 'index columns'.")
        datacols = set(columns) & set(self.datacols)
        if datacols and not skip_data:
            self.table.drop(
                columns=datacols,
                inplace=True
            )
        indexcols = set(columns) & set(self.indexcols)
        if indexcols and not skip_index:
            index = self.table.index.to_frame()
            index.drop(
                columns=indexcols,
                inplace=True
            )
            index = pd.MultiIndex.from_frame(index)
            if not index.is_unique:
                raise PuffbirdError(f"Dropping index columns '{indexcols}' "
                                    "results in non-unique indices.")
            self.table.index = index
        return self

    def rename(
        self,
        **rename_kws: str
    ):
        """
        Rename columns in place.

        Parameters
        ----------
        rename_kws : dict

        Returns
        -------
        self

        See Also
        --------
        pandas.DataFrame.rename
        """

        self.table.rename(
            columns=rename_kws,
            inplace=True
        )
        self.table.rename_axis(
            index=rename_kws,
            inplace=True
        )
        return self

    def to_puffy(
        self,
        *indexcols: str,
        keep_missing_idcs: bool = True,
        aggfunc: Union[callable, Dict[str, callable]] = DEFAULT_AGGFUNC,
        dropna: bool = True,
        inplace: bool = False
    ):
        """
        Make the `table` *"puffier"* by aggregating across unique sets of
        *"index columns"*

        Parameters
        ----------
        indexcols : str
        keep_missing_idcs : bool, optional
        aggfunc : callable or dict of callables, optional
        dropna : bool, optional
        inplace : bool, optional

        Returns
        -------
        :obj:`~pandas.DataFrame`
        """
        # TODO shared axes? - proper long to puffy method

        if keep_missing_idcs:
            table = self.table.reset_index(
                level=list(set(self.indexcols)-set(indexcols))
            )
        elif not inplace:
            table = self.table.copy()
        # drop nanas
        if dropna:
            table.dropna(inplace=True)

        aggfunc, default_aggfunc = _mapping_variable_converter(
            table, aggfunc, DEFAULT_AGGFUNC, "aggfunc")

        def helper_func(df):
            # convert to dictionary
            dictionary = df.to_dict("list")
            # only loop if aggfunc is not empty
            # or the default method is not list
            if aggfunc or (default_aggfunc != list):
                for key, value in dictionary.items():
                    method = aggfunc.get(key, default_aggfunc)
                    dictionary[key] = method(value)
            # convert to pandas Series
            return pd.Series(dictionary)
        # aggregate
        return table.groupby(list(indexcols)).aggregate(helper_func)

    def multid_pivot(self, values=None, *dims):
        """
        Not Yet Implemented!
        """
        # TODO long frame to xarray? - multidimensional pivot
        raise NotImplementedError("multid_pivot")


# --- helper functions ---


def _col_no_match(datacol, key):
    return re.match(DATACOL_REGEX.format(datacol=datacol), key) is None


def _get_col_name(datacol, n, shared_axes):
    # if it is a shared axes the column name is key
    # else it is "datacol_n"
    for key, shared in shared_axes.items():
        if shared.get(datacol, None) == n:
            return key
    return f"{datacol}_level{n}"


def _select_frame(table, col):
    if col in table.columns:
        return table[col]
    else:
        table.index.to_frame()[col]


def _label_character_replacement(label):
    return label.strip(
        ''
    ).replace(
        '#', '_'
    ).replace(
        '-', '_'
    ).replace(
        '@', 'at'
    ).replace(
        '(', '_'
    ).replace(
        ')', '_'
    ).replace(
        ' ', '_'
    ).replace(
        '"', ''
    ).replace(
        "'", ''
    ).replace(
        "`", ''
    ).replace(
        "%", "perc"
    ).replace(
        '$', 'Dollar'
    ).replace(
        '&', '_and_'
    ).replace(
        '*', '_x_'
    ).replace(
        ',', '_'
    ).replace(
        ';', '_'
    ).replace(
        ':', '_'
    ).replace(
        '.', '_'
    ).replace(
        '?', '_'
    )


def _mapping_variable_converter(table, arg, default_arg, name):
    if isinstance(arg, dict):
        arg = arg.copy()
        default_arg = arg.pop(
            "_default", default_arg
        )
        remaining = set(arg) - set(table.columns)
        if remaining:
            raise PuffbirdError(f"The '{name}' dictionary "
                                "contains keys that are not in "
                                f"the columns {tuple(table.columns)}: "
                                f"'{remaining}'")
    else:
        default_arg = arg
        arg = {}
    return arg, default_arg


def _process_table_when_series(table):
    if table.name is None:
        if "data_column" in table.index.names:
            raise PuffbirdError("When table is a pandas.Series "
                                "object, the index names cannot "
                                "contain the name 'data_column'.")
        return table.to_frame(name='data_column')
    return table.to_frame()


def _process_table_when_unknown_object(table):
    try:
        return pd.DataFrame(table)
    except Exception as e:
        raise PuffbirdError("Cannot convert 'table' argument of type "
                            f"'{type(table)}' to dataframe: {e}")


def _process_column_types(table, datacols, indexcols):
    if datacols is None and indexcols is None:
        # indexcols already in index
        return table, datacols, indexcols

    if indexcols is None:
        indexcols = list(set(table.columns) - set(datacols))
    elif datacols is None:
        datacols = list(set(table.columns) - set(indexcols))

    # no columns given that are not in dataframe
    truth = set(datacols) - set(table.columns)
    assert not truth, (
        f"datacols contains columns not in dataframe: {truth}."
    )
    truth = set(indexcols) - set(table.columns)
    assert not truth, (
        f"indexcols contains columns not in dataframe: {truth}."
    )
    # keep original index for uniqueness
    if not indexcols:
        pass
    else:
        table.set_index(indexcols, append=True, inplace=True)

    # if only a few columns were selected
    if set(table.columns) - set(datacols):
        table = table[datacols]

    # return
    return table, datacols, indexcols


def _enforce_identifier_column_types(
    table, handle_column_types, enforce_identifier_string
):
    # if not handling column types
    if not handle_column_types:
        return table

    # convert data columns
    datacols_rename = {}
    for datacol in table.columns:
        if isinstance(datacol, tuple):
            if not enforce_identifier_string:
                new_datacol = ", ".join(datacol)
            elif all(str(idata).isdigit() for idata in datacol):
                new_datacol = "data_tuple_col_" + "_".join(datacol)
            else:
                # replace various characters
                new_datacol = _label_character_replacement("_".join(datacol))
        elif isinstance(datacol, str):
            if not enforce_identifier_string:
                new_datacol = datacol
            elif datacol.isdigit():
                new_datacol = "data_col_" + datacol
            else:
                # replace various characters
                new_datacol = _label_character_replacement(datacol)
        elif isinstance(datacol, int):
            new_datacol = f"data_col_{datacol}"
        else:
            raise PuffbirdError("Datacolumn must string or integer "
                                f"but is type: {type(datacol)}.")

        if datacol != new_datacol:
            datacols_rename[datacol] = new_datacol

    # rename columns
    if datacols_rename:
        table.rename(columns=datacols_rename, inplace=True)

    # convert index columns
    indexcols_rename = {}
    for idx, indexcol in enumerate(table.index.names):
        if isinstance(indexcol, tuple):
            if not enforce_identifier_string:
                new_indexcol = ", ".join(indexcol)
            elif all(str(idx).isdigit() for idx in indexcol):
                new_indexcol = "index_tuple_col_" + "_".join(indexcol)
            else:
                # replace various characters
                new_indexcol = _label_character_replacement("_".join(indexcol))
        elif indexcol is None:
            new_indexcol = "index_new_col_" + idx
        elif isinstance(indexcol, str):
            if not enforce_identifier_string:
                new_indexcol = indexcol
            elif indexcol.isdigit():
                new_indexcol = "index_col_" + indexcol
            else:
                # replace various characters
                new_indexcol = _label_character_replacement(indexcol)
        elif isinstance(indexcol, int):
            new_indexcol = f"index_col_{indexcol}"
        else:
            raise PuffbirdError("Indexcolumn must string or integer "
                                f"but is type: {type(indexcol)}.")

        if indexcol != new_indexcol:
            indexcols_rename[indexcol] = new_indexcol

    # rename indices
    if indexcols_rename:
        table.rename_axis(index=indexcols_rename, inplace=True)

    return table


def _check_table_column_types(table, enforce_identifier_string):
    # columns and index names must be identifier string types
    for datacol in table.columns:
        if not isinstance(datacol, str):
            raise PuffbirdError(f"Datacolumn '{datacol}' is not a "
                                f"string type: {type(datacol)}")
        if not datacol.isidentifier() and enforce_identifier_string:
            raise PuffbirdError(f"Datacolumn '{datacol}' is not a "
                                "identifier string type.")
    if len(set(table.columns)) != len(table.columns):
        raise PuffbirdError(f"Datacols '{tuple(table.columns)}' "
                            "are not unique.")
    for indexcol in table.index.names:
        if not isinstance(indexcol, str):
            raise PuffbirdError(f"Indexcolumn '{indexcol}' is not a "
                                f"string type: {type(indexcol)}")
        if not indexcol.isidentifier() and enforce_identifier_string:
            raise PuffbirdError(f"Indexcolumn '{indexcol}' is not a"
                                "identifier string type.")
        for datacol in table.columns:
            if not _col_no_match(datacol, indexcol):
                raise PuffbirdError(f"Indexcolumn '{indexcol}' matches "
                                    f"datacol '{datacol}': Indexcol "
                                    "cannot start the same way as "
                                    "datacol.")
    if len(set(table.index.names)) != len(table.index.names):
        raise PuffbirdError(f"Indexcols '{tuple(table.index.names)}' "
                            f"are not unique.")


def _check_shared_axes_argument(shared_axes, datacols, indexcols):
    for key, shared in shared_axes.items():
        # must be dictionary
        if not isinstance(shared, dict):
            raise PuffbirdError("All shared axes arguments must be "
                                "dictionaries, but the value for key "
                                f"'{key}' is of type '{type(shared)}'")
        # keys must be in columns
        not_in_columns = set(shared) - set(datacols)
        if not_in_columns:
            raise PuffbirdError("All keys of the dictionary of a shared "
                                "axes argument must be present in the "
                                f"'datacols' {datacols}; these keys "
                                f"are not in columns: '{not_in_columns}'.")
        # keys must be unique
        key_is_unique = (
            all(_col_no_match(datacol, key)
                for datacol in datacols)
            and key not in indexcols
        )
        if not key_is_unique:
            raise PuffbirdError(f"The keyword '{key}' is not unique: "
                                "It must not exist in in the 'datacols' "
                                f"{datacols} or the 'indexcols' "
                                f"{indexcols}, and it cannot start "
                                "the same way as any 'datacolumn' in "
                                "the dataframe.")
