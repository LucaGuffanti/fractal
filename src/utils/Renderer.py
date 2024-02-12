import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

CMAP = "twilight"
INTERPOLATION = "None"
FPS = 30


def render_fractal(fractal):
    ratio = (fractal)


    fig = plt.figure(figsize=(10, 10))

    plt.axis("off")
    plt.imshow(fractal, interpolation=INTERPOLATION, cmap=CMAP)
    plt.gca().set_aspect('auto')
    plt.show()


def render_fractal_batch(fractals):
    for fractal in fractals:
        render_fractal(fractal)


def save_fractal(fractal, name):
    plt.figure(figsize=(30, 30))

    plt.imshow(fractal, interpolation=INTERPOLATION, cmap=CMAP, origin="lower")
    plt.savefig(name)

    plt.show()


def animate_transition(fractals, save=True):
    ims = []
    fig, ax = plt.subplots()

    for i in range(len(fractals)):
        im = ax.imshow(fractals[i], interpolation=INTERPOLATION, cmap=CMAP, animated=True, origin="lower")
        if i == 0:
            im = ax.imshow(fractals[i], interpolation=INTERPOLATION, cmap=CMAP, origin="lower")
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    ani.save("animation.mp4", fps=FPS, dpi=300)
    plt.show()
