import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import iirfilter, filtfilt


init_params = {
    'amplitude': 1.0,
    'frequency': 0.3,
    'phase': 0.0,
    'noise_mean': 0.0,
    'noise_cov': 0.1,
    'cutoff_freq': 5.0
}

t = np.linspace(0, 10, 1000)
stored_noise = None

def harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_cov, noise=None):
    y_clean = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if noise is None:
        noise = np.random.normal(noise_mean, np.sqrt(noise_cov), size=t.shape)
    y_noisy = y_clean + noise
    return y_clean, y_noisy, noise

def apply_filter(y, fs, cutoff, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = iirfilter(order, normal_cutoff, btype='low', ftype='butter')
    return filtfilt(b, a, y)

y_clean, y_noisy, stored_noise = harmonic_with_noise(
    t,
    amplitude=init_params['amplitude'],
    frequency=init_params['frequency'],
    phase=init_params['phase'],
    noise_mean=init_params['noise_mean'],
    noise_cov=init_params['noise_cov']
)
fs = len(t) / (t[-1] - t[0])

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.5)
line_clean, = ax.plot(t, y_clean, 'b--', label='Clean')
line_filtered, = ax.plot(t, apply_filter(y_noisy, fs, init_params['cutoff_freq']), 'purple', label='Filtered')
line_noisy, = ax.plot(t, y_noisy, 'orange', alpha=0.6, label='Noisy')
ax.set_ylim(-2, 2)
ax.legend()

slider_names = {
    'amplitude': (0.1, 2.0),
    'frequency': (0.01, 1.0),
    'phase': (0.0, 2 * np.pi),
    'noise_mean': (-1.0, 1.0),
    'noise_cov': (0.001, 1.0),
    'cutoff_freq': (0.1, 10.0)
}
sliders = {}
for i, (name, (vmin, vmax)) in enumerate(slider_names.items()):
    ax_slider = plt.axes([0.25, 0.45 - i * 0.05, 0.65, 0.03])
    sliders[name] = Slider(ax_slider, name.replace('_', ' ').title(), vmin, vmax, valinit=init_params[name])

check_ax = plt.axes([0.8, 0.05, 0.15, 0.15])
checkbox = CheckButtons(check_ax, ['Show Noise', 'Show Filtered'], [True, True])

reset_ax = plt.axes([0.1, 0.05, 0.1, 0.04])
button = Button(reset_ax, 'Reset')

def update(val):
    global stored_noise

    a = sliders['amplitude'].val
    f = sliders['frequency'].val
    ph = sliders['phase'].val
    m = sliders['noise_mean'].val
    c = sliders['noise_cov'].val
    cutoff = sliders['cutoff_freq'].val

    if val in (sliders['amplitude'], sliders['frequency'], sliders['phase'], sliders['cutoff_freq']):
        y_clean = a * np.sin(2 * np.pi * f * t + ph)
        y_noisy = y_clean + stored_noise
    else:
        y_clean, y_noisy, stored_noise = harmonic_with_noise(t, a, f, ph, m, c)

    y_filtered = apply_filter(y_noisy, fs, cutoff)

    line_clean.set_ydata(y_clean)
    line_noisy.set_ydata(y_noisy)
    line_filtered.set_ydata(y_filtered)
    line_noisy.set_visible(checkbox.get_status()[0])
    line_filtered.set_visible(checkbox.get_status()[1])

    fig.canvas.draw_idle()

for s in sliders.values():
    s.on_changed(update)

def toggle_visibility(label):
    line_noisy.set_visible(checkbox.get_status()[0])
    line_filtered.set_visible(checkbox.get_status()[1])
    fig.canvas.draw_idle()

checkbox.on_clicked(toggle_visibility)

def reset(event):
    global stored_noise
    for s in sliders.values():
        s.reset()
    stored_noise = None
    update(None)

button.on_clicked(reset)

plt.show()
