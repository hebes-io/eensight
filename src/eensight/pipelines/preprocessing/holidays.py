# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebook/prophet

import holidays as hdays
import pandas as pd
from unidecode import unidecode

from .validation import merge_on_days


def make_holidays_df(year_list, country, province=None, state=None):
    """Make dataframe of holidays for given years and countries

    Args:
        year_list: a list of years
        country: country name

    Returns:
        Dataframe with 'timestamp' and 'holiday' features
    """

    try:
        holidays = getattr(hdays, country)(
            prov=province, state=state, years=year_list, expand=False
        )
    except AttributeError as e:
        raise AttributeError(
            f"Holidays in {country} are not currently supported!"
        ) from e

    holidays_df = pd.DataFrame(
        data=[(date, holidays.get_list(date)) for date in holidays],
        columns=["timestamp", "holiday"],
    )
    holidays_df = holidays_df.explode("holiday")
    holidays_df["holiday"] = holidays_df["holiday"].map(unidecode)
    holidays_df = holidays_df.set_index("timestamp")
    return holidays_df


def add_holidays(X, country, province=None, state=None):
    """Add a holiday feature column to an existing dataframe."""
    year_list = X.groupby(lambda x: x.year).first().index.tolist()
    holidays = make_holidays_df(
        year_list, country=country, province=province, state=state
    )

    if "holiday" in X.columns:
        raise ValueError(
            "Cannot add a holiday column since a column "
            "named `holiday` alreday exists"
        )
    else:
        return merge_on_days(X, holidays)
