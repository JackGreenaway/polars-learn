import polars as pl
import pytest


@pytest.fixture
def frame():
    return pl.DataFrame({"price": [1.0, 2.0, 3.0, 4.0], "quantity": [2, 3, 4, 5]})
