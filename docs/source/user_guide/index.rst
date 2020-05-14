{{ header }}

.. _user_guide:

==========
User Guide
==========

Installation
------------

`Puffbird` can be installed via pip from `PyPI <https://pypi.org/project/puffbird>`__:

.. code-block:: bash

   pip install puffbird

You can also clone the git repository and install the package from source.


Quick Start
-----------

The main functionality that `puffbird` adds to `pandas`
is the ability to easily **"explode"** *"puffy"* tables:

.. ipython:: python

    import pandas as pd
    import puffbird as pb
    df = pd.DataFrame({
        'a': [[1,2,3], [4,5,6,7], [3,4,5]],
        'b': [{'c':['asdf'], 'd':['ret']}, {'d':['r']}, {'c':['ff']}],
     })
     df

As you can see, this dataframe is *"puffy"*, it has various non-hashable
object types that can be iterated over. To quickly create a *long-format*
:obj:`~pandas.DataFrame`, we can use the :obj:`puffy_to_long` function:

.. ipython:: python

    long_df = pb.puffy_to_long(df)
    long_df


Tutorials
---------

* `Creating long dataframe from puffy tables <intro_tolong.ipynb>`_.


.. toctree::
    :maxdepth: 2
    :glob:
    :hidden:

    philosophy
    From "puffy" to long-format <intro_tolong.nblink>
