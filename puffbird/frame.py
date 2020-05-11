"""
FrameEngine class
-----------------
Class to transform a wide pandas DataFrame, as it may be obtained from a
database model like datajoint, into a long dataframe format optimal for
plotting and groupby computations.
"""

import re

import pandas as pd

from .err import PuffbirdError

RESERVED_COLUMNS = {'applyfunc_result', 'max_depth', 'dropna', 'transformer'}
DEFAULT_MAX_DEPTH = 3
DEFAULT_REDUCE_METHOD = list
DATACOL_REGEX = '{datacol}(_level)?[1-9]*$'

# TODO merging tables?
# TODO use maybe wide_to_long?
# TODO other iter functions (with dictionaries etc.)
# TODO collapse class or within transformer
# TODO make identifier columns from other column names
# TODO transformer for long dataframes
# TODO relax identifier restriction for columns?
# TODO index to multiindex when only single index?
# TODO long frame to xarray?


class FrameEngine:
    """class to transform wide dataframes pulled from datajoint

    Parameters
    ----------
    table : pandas.DataFrame
        A dataframe with columns defining datatypes and rows being different
        entries. All column and index names must be identifier string types.
        Individual cells in the dataframe may have arbitrary objects in them.
    datacols : list-like
        The columns in the dataframe that are considered "data"; i.e. for
        example columns where each cell is a numpy.array, e.g. a timestamp
        array. Defaults to None.
    indexcols : list-like
        The columns in the dataframe that are immutable types, e.g. strings or
        integers. Defaults to None.
    inplace : bool
        If possible do not copy dataframe. Defaults to False.

    Notes
    -----
    Something something.

    Examples
    --------
    >>> pb.FrameEngine(df)
    """

    def __init__(
        self,
        table,
        datacols=None,
        indexcols=None,
        inplace: bool = False,
        handle_column_types: bool = True,
        enforce_identifier_string: bool = True
    ):
        if isinstance(table, pd.Series):
            table = self._process_table_when_series(table)
        elif not isinstance(table, pd.DataFrame):
            table = self._process_table_when_unknown_object(table)

        truth = RESERVED_COLUMNS & set(table.columns)
        if truth:
            raise PuffbirdError(f"Dataframe table has columns "
                                f"that are reserved: {truth}")

        if not inplace:
            table = table.copy()

        table, datacols, indexcols = self._process_column_types(
            table, datacols, indexcols
        )

        # table index must be a multiindex
        if not isinstance(table.index, pd.MultiIndex):
            table.index = pd.MultiIndex.from_frame(table.index.to_frame())

        table = self._enforce_identifier_column_types(
            table, handle_column_types, enforce_identifier_string
        )

        # check table index and column types
        self._check_table_column_types(table, enforce_identifier_string)

        # check if index is unique
        if not table.index.is_unique:
            raise PuffbirdError("Each row for all index columns "
                                "must be a unique set.")

        # assign table
        self._table = table

    @staticmethod
    def _process_table_when_series(table):
        if table.name is None:
            if 'data_column' in table.index.names:
                raise PuffbirdError("When table is a pandas.Series "
                                    "object, the index names cannot "
                                    "contain the name 'data_column'.")
            return table.to_frame(name='data_column')
        return table.to_frame()

    @staticmethod
    def _process_table_when_unknown_object(table):
        try:
            return pd.DataFrame(table)
        except Exception as e:
            raise PuffbirdError(f"Cannot convert 'table' argument of type "
                                f"'{type(table)}' to dataframe: {e}")

    @staticmethod
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
            f'datacols contains columns not in dataframe: {truth}.'
        )
        truth = set(indexcols) - set(table.columns)
        assert not truth, (
            f'indexcols contains columns not in dataframe: {truth}.'
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
        self, table, handle_column_types, enforce_identifier_string
    ):
        # if not handling column types
        if not handle_column_types:
            return table

        # convert data columns
        datacols_rename = {}
        for datacol in table.columns:
            if isinstance(datacol, str):
                if not enforce_identifier_string:
                    new_datacol = datacol
                elif datacol.isdigit():
                    new_datacol = 'data_column_' + datacol
                else:
                    # replace various characters
                    new_datacol = self._label_character_replacement(datacol)
            elif isinstance(datacol, int):
                new_datacol = f'data_column_{datacol}'
            else:
                raise PuffbirdError(f"Datacolumn must string or integer "
                                    f"but is type: {type(datacol)}.")

            if datacol != new_datacol:
                datacols_rename[datacol] = new_datacol

        # rename columns
        if datacols_rename:
            table.rename(columns=datacols_rename, inplace=True)

        # convert index columns
        indexcols_rename = {}
        for idx, indexcol in enumerate(table.index.names):
            if indexcol is None:
                new_indexcol = 'index_column_' + idx
            elif isinstance(indexcol, str):
                if not enforce_identifier_string:
                    new_indexcol = indexcol
                elif indexcol.isdigit():
                    new_indexcol = 'index_column_' + indexcol
                else:
                    # replace various characters
                    new_indexcol = self._label_character_replacement(indexcol)
            elif isinstance(indexcol, int):
                new_indexcol = f'index_column_{indexcol}'
            else:
                raise PuffbirdError(f"Indexcolumn must string or integer "
                                    f"but is type: {type(indexcol)}.")

            if indexcol != new_indexcol:
                indexcols_rename[indexcol] = new_indexcol

        # rename indices
        if indexcols_rename:
            table.rename_axis(index=indexcols_rename, inplace=True)

        return table

    @staticmethod
    def _label_character_replacement(label):
        return label.replace(
            '#', '_'
        ).replace(
            '-', '_'
        ).replace(
            '@', 'at'
        ).replace(
            '(', '_'
        ).replace(
            ')', '_'
        ).strip(
            ''
        ).replace(
            '', '_'
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
        )

    def _check_table_column_types(self, table, enforce_identifier_string):
        # columns and index names must be identifier string types
        for datacol in table.columns:
            if not isinstance(datacol, str):
                raise PuffbirdError(f"Datacolumn '{datacol}' is not a "
                                    f"string type: {type(datacol)}")
            if not datacol.isidentifier() and enforce_identifier_string:
                raise PuffbirdError(f"Datacolumn '{datacol}' is not a "
                                    f"identifier string type.")
        if len(set(table.columns)) != len(table.columns):
            raise PuffbirdError(f"Datacols '{tuple(table.columns)}' "
                                f"are not unique.")
        for indexcol in table.index.names:
            if not isinstance(indexcol, str):
                raise PuffbirdError(f"Indexcolumn '{indexcol}' is not a "
                                    f"string type: {type(indexcol)}")
            if not indexcol.isidentifier() and enforce_identifier_string:
                raise PuffbirdError(f"Indexcolumn '{indexcol}' is not a"
                                    f"identifier string type.")
            for datacol in table.columns:
                if self._datacol_rematch(datacol, indexcol):
                    raise PuffbirdError(f"Indexcolumn '{indexcol}' matches "
                                        f"datacol '{datacol}': Indexcol "
                                        f"cannot start the same way as "
                                        f"datacol.")
        if len(set(table.index.names)) != len(table.index.names):
            raise PuffbirdError(f"Indexcols '{tuple(table.index.names)}' "
                                f"are not unique.")

    @property
    def table(self):
        """
        Puffy `pandas.DataFrame` passed during initialization.
        """
        return self._table

    @property
    def datacols(self):
        """
        Tuple of data columns in the dataframe.
        """
        return tuple(self._table.columns)

    @property
    def indexcols(self):
        """
        Tuple of index columns in the dataframe.
        """
        return tuple(self._table.index.names)

    def tolong(
        self, *,
        transformer: callable = iter,
        max_depth=DEFAULT_MAX_DEPTH,
        dropna: bool = True,
        **shared_axes
    ) -> "pd.DataFrame":
        """
        Transform the dataframe into a long format dataframe.

        Parameters
        ----------
        transformer : callable, optional
            function called on each cell for each "data column" to create
            a new pandas.Series. If the "data columns" only contain array-like
            objects the default function `iter` is sufficient. If the
            "data columns" also contain other objects such as dictionaries,
            it may be necessary to provide a custom callable.
        max_depth : int, list or dict, optional
            Maximum depth of expanding each cell, before the algorithm stops
            for each "data column". If we set the max_depth to 3, for example,
            a "data column" consisting of 4-D numpy.arrays will result in a
            long dataframe where the "data column" cells contain
            1-D numpy.arrays. If the arrays were 3-D, it will result in a
            long dataframe with floats/ints in each cell. Defaults to 3.
        dropna : bool, optional
            Drop rows in long dataframe where all "data columns" are NaN.
        shared_axes : dict, optional
            Specify if two or more "data columns" share axes. The keyword
            will correspond to what the column will be called in the long
            dataframe. Each argument is a dictionary where the keys
            correspond to the names of the "data columns", which share
            an axis, and the value correspond to the depth/axis is shared
            for each "data column".
        """
        # TODO simply a shortcut tolong

        # check shared axes arguments for correct formatting
        self._check_shared_axes_argument(shared_axes)

        # convert max_depth correctly
        max_depth, default_max_depth = self._max_depth_converter(max_depth)

        # iterate of each data column
        for m, (datacol, series) in enumerate(self.table.items()):
            # set first depth
            n = 0
            # if series already not object skip
            while (
                series.dtype == object
                and max_depth.get(datacol, default_max_depth) > n
            ):
                # superstack pandas.Series object
                series = self._superstack_series(
                    series, datacol, transformer, dropna,
                    self._get_col_name(datacol, n, shared_axes)
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
                df = pd.merge(df, _df, on=on, how='outer')

        return df

    def _max_depth_converter(self, max_depth):
        if isinstance(max_depth, dict):
            max_depth = max_depth.copy()
            default_max_depth = max_depth.pop('_default', DEFAULT_MAX_DEPTH)
            remaining = set(max_depth) - set(self.datacols)
            if remaining:
                raise PuffbirdError(f"The 'max_depth' dictionary "
                                    f"contains keys that are not in "
                                    f"'datacols' {self.datacols}: "
                                    f"'{remaining}'")
        if isinstance(max_depth, list):
            if len(max_depth) != len(self.datacols):
                raise PuffbirdError(f"The 'max_depth' list is not the same "
                                    f"length as 'datacols': {len(max_depth)}"
                                    f"!={len(self.datacols)}.")
            max_depth = {
                datacol: max_depth_ele
                for datacol, max_depth_ele in zip(self.datacols, max_depth)
            }
            default_max_depth = DEFAULT_MAX_DEPTH
        else:
            default_max_depth = max_depth
            max_depth = {}
        return max_depth, default_max_depth

    def _check_shared_axes_argument(self, shared_axes):
        for key, shared in shared_axes.items():
            # must be dictionary
            if not isinstance(shared, dict):
                raise PuffbirdError(f"All shared axes arguments must be "
                                    f"dictionaries, but the value for key "
                                    f"'{key}' is of type '{type(shared)}'")
            # keys must be in columns
            not_in_columns = set(shared) - set(self.datacols)
            if not_in_columns:
                raise PuffbirdError(f"All keys of the dictionary of a shared "
                                    f"axes argument must be present in the "
                                    f"'datacols' {self.datacols}; these keys "
                                    f"are not in columns: '{not_in_columns}'.")
            # keys must be unique
            key_is_unique = (
                all(self._datacol_rematch(datacol, key)
                    for datacol in self.table.columns)
                and key not in self.table.index.names
            )
            if not key_is_unique:
                raise PuffbirdError(f"The keyword '{key}' is not unique: "
                                    f"It must not exist in in the 'datacols' "
                                    f"{self.datacols} or the 'indexcols' "
                                    f"{self.indexcols}, and it cannot start "
                                    f"the same way as any 'datacolumn' in "
                                    f"the dataframe.")

    @staticmethod
    def _datacol_rematch(datacol, key):
        return re.match(DATACOL_REGEX.format(datacol=datacol), key) is None

    @staticmethod
    def _get_col_name(datacol, n, shared_axes):
        # if it is a shared axes the column name is key
        # else it is "datacol_n"
        for key, shared in shared_axes.items():
            if shared.get(datacol, None) == n:
                return key
        return f"{datacol}_level{n}"

    @staticmethod
    def _superstack_series(series, datacol, transformer, dropna, col_name):
        # apply series transformer (iter is default for sequences)
        # series.index is already assumed to be multi index
        # transform into dataframe
        # this should automatically infer types
        table = series.apply(
            lambda x: pd.Series(transformer(x)),
            convert_dtype=True
        )
        # give columns index a name
        table.columns.name = col_name
        # stack dataframe
        if isinstance(table.columns, pd.MultiIndex):
            levels = list(range(table.columns.nlevels))
            series = table.stack(levels=levels, dropna=dropna)
        else:
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
            return self.__class__(selected_table)
        else:
            return selected_table

    def cols_tolong(self, *cols, **kwargs):
        """
        Same as `tolong` but only applied to specific columns.
        """
        return self[list(cols)].tolong(**kwargs)

    def expand_col(self, col, reset_index=True):
        """
        Expand a column that contains long dataframes and return
        a single long dataframe.
        """

        series = self[col]
        long_df = pd.concat(
            list(series), keys=series.index, names=series.index.names,
            sort=False
        )

        if reset_index:
            return long_df.reset_index()
        else:
            return long_df

    def col_apply(
        self, func, col, new_col_name=None, assign_to_index=None, **kwargs
    ):
        """apply a function to entries of a single column

        Parameters
        ----------
        func : callable
            Function to apply.
        column : str
            Name of column
        new_col_name : str, optional
            Name of computed new column. If None, this will be set
            to the name of the column; i.e. the name of the column will be
            overwritter. Defaults to None.
        assign_to_index : bool, optional
            Assign new column as 'index column' instead of as 'data column'.
        kwargs : dict
            Keyword Arguments passed to the apply method of a pandas.Series,
            and thus to the function.
        """
        if new_col_name is None:
            new_col_name = col
        if assign_to_index is None:
            assign_to_index = col in self.indexcols
        # apply function
        series = self._select_frame(self.table, col).apply(func, **kwargs)
        # assign output
        self._assign_output_series(series, new_col_name, assign_to_index)
        return self

    def apply(
        self, func, new_col_name, *args,
        assign_to_index=False, map_kws={}, **kwargs
    ):
        """
        Apply a function across columns by mapping args and kwargs of func.

        Parameters
        ----------
        func : callable
            Function to apply.
        new_col_name : str
            Name of computed new col. If None, new_col_name will be
            called 'applyfunc_result'.
        args : tuple
            Arguments passed to function. Each argument should be a column
            in the dataframe. This value is passed instead of the string.
        assign_to_index : bool, optional
            Assign new column as 'index column' instead of as 'data column'.
        map_kws : dict
            Same as args just as keyword arguments.
        kwargs: dict
            Keyword arguments passed to function as is.
        """
        if new_col_name is None:
            new_col_name = 'applyfunc_result'
        # apply function
        series = self.table.reset_index().apply(
            lambda x: func(
                *(x[arg] for arg in args),
                **{key: x[arg] for key, arg in map_kws.items()},
                **kwargs
            ),
            axis=1, result_type='reduce'
        )
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
                                    f"assigned to index columns; cannot "
                                    f"assign to data columns. Choose "
                                    f"different name.")
            self.table[new_col_name] = series

    @staticmethod
    def _select_frame(table, col):
        if col in table.columns:
            return table[col]
        else:
            table.index.to_frame()[col]

    def drop(self, *columns, skip=False, skip_index=False, skip_data=False):
        """Drop columns in place.
        """
        not_found = set(columns) - (set(self.datacols) | set(self.indexcols))
        if not_found and not skip:
            raise PuffbirdError(f"Columns '{not_found}' are not in "
                                f"'data columns' or 'index columns'.")
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
                                    f"results in non-unique indices.")
            self.table.index = index
        return self

    def rename(self, **rename_kws):
        """
        Rename columns in place.
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

    def reduce(
        self, *indexcols,
        reduce_method=DEFAULT_REDUCE_METHOD, dropna=True
    ):
        """
        Reduce to a new unique set of indexcolumns by collapsing/aggregating
        the existing entries.
        """
        # TODO handling nans?
        reduce_method, default_reduce_method = self._reduce_method_converter(
            reduce_method
        )

        def helper_func(df):
            if dropna:
                df.dropna(inplace=True)
            dictionary = df.to_dict('list')
            for key, method in reduce_method.items():
                dictionary[key] = method(dictionary[key])
            return pd.Series(dictionary)

        return self.table.groupby(list(indexcols)).aggregate(helper_func)

    def _reduce_method_converter(self, reduce_method):
        if isinstance(reduce_method, dict):
            reduce_method = reduce_method.copy()
            default_reduce_method = reduce_method.pop(
                '_default', DEFAULT_REDUCE_METHOD
            )
            remaining = set(reduce_method) - set(self.datacols)
            if remaining:
                raise PuffbirdError(f"The 'reduce_method' dictionary "
                                    f"contains keys that are not in "
                                    f"'datacols' {self.datacols}: "
                                    f"'{remaining}'")
        if isinstance(reduce_method, list):
            if len(reduce_method) != len(self.datacols):
                raise PuffbirdError(f"The 'reduce_method' list is not the "
                                    f"same length as 'datacols': "
                                    f"{len(reduce_method)}"
                                    f"!={len(self.datacols)}.")
            reduce_method = {
                datacol: reduce_method_ele
                for datacol, reduce_method_ele
                in zip(self.datacols, reduce_method)
            }
            default_reduce_method = DEFAULT_REDUCE_METHOD
        else:
            default_reduce_method = reduce_method
            reduce_method = {}
        return reduce_method, default_reduce_method
