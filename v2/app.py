from datetime import datetime

import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts

pn.extension()
hv.extension('bokeh')


# Część odpowiedzialna za heatmapę
def generate_matrix(rows=10, columns=10):
    return np.random.rand(rows, columns)


def heatmap():
    matrix = generate_matrix()
    heatmap = hv.HeatMap((np.array([i for j in range(matrix.shape[1]) for i in range(matrix.shape[0])]),
                          np.array([j for j in range(matrix.shape[1]) for i in range(matrix.shape[0])]),
                          matrix.flatten()))
    return heatmap.opts(width=500, height=500, tools=['hover'], colorbar=True)


# Część odpowiedzialna za szereg czasowy
initial_time = datetime.now()
times = [initial_time]
values = [np.random.rand()]


def generate_time_series_value():
    current_time = datetime.now()
    times.append(current_time)
    values.append(np.random.rand())
    return hv.Curve((times, values)).opts(width=500, height=500, tools=['hover'])


time_series = pn.pane.HoloViews(generate_time_series_value())


# Aktualizacja szeregu czasowego co pewien czas
def update_time_series():
    time_series.object = generate_time_series_value()


pn.state.add_periodic_callback(update_time_series, period=1000)  # Aktualizacja co 1000ms (1 sekunda)

# Tworzymy panel z przyciskiem, który generuje heatmapę
btn = pn.widgets.Button(name='Generuj Heatmapę')
# btn = pn.widgets.Button(name='Generuj Heatmapę', button_type='primary')
heatmap_pane = pn.pane.HoloViews(heatmap())

MAX_POINTS = 1000
x_values = list(range(MAX_POINTS))
y_values = [None] * MAX_POINTS
current_index = 0


def generate_fixed_domain_time_series():
    global current_index

    # Dodajemy nową wartość tylko jeśli nie przekroczyliśmy maksymalnej liczby punktów
    if current_index < MAX_POINTS:
        y_values[current_index] = np.random.rand()
        current_index += 1

    return hv.Curve((x_values, y_values)).opts(width=500, height=500, tools=['hover'])


fixed_domain_time_series = pn.pane.HoloViews(generate_fixed_domain_time_series())


def update_fixed_domain_time_series():
    fixed_domain_time_series.object = generate_fixed_domain_time_series()


pn.state.add_periodic_callback(update_fixed_domain_time_series, period=50)


@btn.on_click
def update_heatmap(event):
    heatmap_pane.object = heatmap()


app = pn.Column(pn.Row(btn, heatmap_pane), time_series, fixed_domain_time_series)

app.servable()
# if __name__ == "__main__":
