# -*- encoding: utf-8 -*-
"""
H2O Segment Models.

:copyright: (c) 2020 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from __future__ import absolute_import

from h2o.base import Keyed
from h2o.frame import H2OFrame
from h2o.expr import ExprNode
from h2o.expr import ASTId

__all__ = ("H2OSegmentModels", )


class H2OSegmentModels(Keyed):
    """
    Collection of H2O Models built for each input segment.

    H2OFrame is similar to pandas' ``DataFrame``, or R's ``data.frame``. One of the critical distinction is that the
    data is generally not held in memory, instead it is located on a (possibly remote) H2O cluster, and thus
    ``H2OFrame`` represents a mere handle to that data.

    Create a new H2OFrame object, possibly from some other object.

    :param python_obj: object that will be converted to an ``H2OFrame``. This could have multiple types:

        - None: create an empty H2OFrame
        - A list/tuple of strings or numbers: create a single-column H2OFrame containing the contents of this list.
        - A dictionary of ``{name: list}`` pairs: create an H2OFrame with multiple columns, each column having the
          provided ``name`` and contents from ``list``. If the source dictionary is not an OrderedDict, then the
          columns in the H2OFrame may appear shuffled.
        - A list of lists of strings/numbers: construct an H2OFrame from a rectangular table of values, with inner
          lists treated as rows of the table. I.e. ``H2OFrame([[1, 'a'], [2, 'b'], [3, 'c']])`` will create a
          frame with 3 rows and 2 columns, one numeric and one string.
        - A Pandas dataframe, or a Numpy ndarray: create a matching H2OFrame.
        - A Scipy sparse matrix: create a matching sparse H2OFrame.

    :param int header: if ``python_obj`` is a list of lists, this parameter can be used to indicate whether the
        first row of the data represents headers. The value of -1 means the first row is data, +1 means the first
        row is the headers, 0 (default) allows H2O to guess whether the first row contains data or headers.
    :param List[str] column_names: explicit list of column names for the new H2OFrame. This will override any
        column names derived from the data. If the python_obj does not contain explicit column names, and this
        parameter is not given, then the columns will be named "C1", "C2", "C3", etc.
    :param column_types: explicit column types for the new H2OFrame. This could be either a list of types for
        each column, or a dictionary of {column name: column type} pairs. In the latter case you may override
        types for only few columns, and let H2O choose the types of the rest.
    :param na_strings: List of strings in the input data that should be interpreted as missing values. This could
        be given on a per-column basis, either as a list-of-lists, or as a dictionary {column name: list of nas}.
    :param str destination_frame: (internal) name of the target DKV key in the H2O backend.
    :param str separator: (deprecated)

    :example:
    >>> python_obj = [1, 2, 2.5, -100.9, 0]
    >>> frame = h2o.H2OFrame(python_obj)
    >>> frame
    """

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, segment_models_id=None):
        self._segment_models_id = segment_models_id

    @property
    def key(self):
        return self._segment_models_id

    def detach(self):
        self._segment_models_id = None

    def as_frame(self):
        return H2OFrame._expr(expr=ExprNode("segment_models_as_frame", ASTId(self._segment_models_id)))._frame(fill_cache=True)
