"""
Variable selector
"""
import re

from sklearn.compose import make_column_selector

class MakeColumnSelector(make_column_selector): # pylint: disable=too-few-public-methods
    """Extend sklearn.compose.make_column_selector"""
    def __init__(self, pattern=None, dtype_include=None,
                 dtype_exclude=None, pattern_exclude=None):
        super(MakeColumnSelector, self).__init__(
            pattern=pattern, dtype_include=dtype_include,
            dtype_exclude=dtype_exclude)
        self.pattern_exclude = pattern_exclude

    def __call__(self,df):
        cols = super(MakeColumnSelector, self).__call__(df)
        if self.pattern_exclude is not None:
            cols = [c for c in cols if not re.findall(self.pattern_exclude,c)]
        return cols
