import re
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from astrorapid.plot_metrics import plot_metrics

def test_plot_metrics(monkeypatch):
    """Test the basic plotting functionality"""
    class_names = ('A', 'B')
    X_test = np.zeros((500, 50, 4))
    y_test = np.zeros((500, 50, len(class_names) + 1))
    fig_dir = 'some/dir'
    timesX_test = np.ones((500, 50))
    num_ex_vs_time = 10
    orig_lc_test = 10 * [MagicMock()]
    objids_test = 500 * ['obj_id']
    passbands = ('first_band', 'second_band')
    init_day_since_trigger = 20

    model = Mock()
    model.predict = Mock(return_value=np.ones((500, 50, len(class_names) + 1)))
    model.evaluate = Mock(return_value=np.ones(len(class_names)))

    mock_rc = Mock()
    mock_legend = Mock()
    monkeypatch.setattr('matplotlib.rc', mock_rc)
    monkeypatch.setattr('matplotlib.axes.Axes.legend', mock_legend)
    monkeypatch.setattr('numpy.concatenate', Mock(return_value=np.ones(5)))

    with pytest.raises(
            ValueError, match=re.escape("min() arg is an empty sequence")):
        plot_metrics(
            class_names, model, X_test, y_test, fig_dir,
            timesX_test, orig_lc_test, objids_test, passbands,
            num_ex_vs_time, init_day_since_trigger
        )
    mock_legend.assert_called()
