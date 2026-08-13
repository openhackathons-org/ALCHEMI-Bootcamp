"""Accessibility checks for notebook-local figure rendering."""

from __future__ import annotations

import helpers
import matplotlib.pyplot as plt
import pandas as pd


def test_render_figure_embeds_bounded_png_and_escaped_alt_text() -> None:
    training = pd.DataFrame(
        {
            "completed optimizer updates": [1, 2],
            "total loss": [2.0, 1.0],
        }
    )
    validation = pd.DataFrame(
        {
            "completed optimizer updates": [2],
            "total loss": [1.5],
        }
    )
    figure = helpers.plot_toy_history(training, validation)

    rendered = helpers.render_figure(
        figure,
        alt_text='Loss < validation & "training"',
    )
    plt.close(figure)

    assert 'src="data:image/png;base64,' in rendered.data
    assert 'alt="Loss &lt; validation &amp; &quot;training&quot;"' in rendered.data
    assert "max-width:100%" in rendered.data
