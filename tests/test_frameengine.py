#!/usr/bin/env python

"""Tests for `puffbird` package."""

import pytest
import xarray as xr
import pandas as pd
import numpy as np


from puffbird.frame import FrameEngine


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    

@pytest.fixture
def test_data():
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C': [1, 2, 3, 4, 5, 6, 7, 8],
        'D': [10, 20, 30, 40, 50, 60, 70, 80]
    })
    return FrameEngine(df)

def test_multid_pivot_data_array(test_data):
    # Test creating a DataArray with two dimensions
    expected_coords = {
        'A': ['bar', 'foo'],
        'B': ['one', 'three', 'two']
    }
    expected_values = np.array([[20., np.nan, 50.], [np.nan, 80., 30.]])
    data_array = test_data.multid_pivot('D', 'A', 'B')
    assert isinstance(data_array, xr.DataArray)
    assert data_array.coords == expected_coords
    assert np.array_equal(data_array.values, expected_values)

def test_multid_pivot_dataset(test_data):
    # Test creating a Dataset with one dimension
    expected_coords = {'A': ['bar', 'foo']}
    expected_values = pd.DataFrame({
        ('C', 'bar'): [2, 4, 6],
        ('C', 'foo'): [1, 7, 8],
        ('D', 'bar'): [20, 40, 60],
        ('D', 'foo'): [10, 70, 80]
    })
    dataset = test_data.multid_pivot(['C', 'D'], 'A')
    assert isinstance(dataset, xr.Dataset)
    assert dataset.coords == expected_coords
    assert dataset.to_dataframe().equals(expected_values)
