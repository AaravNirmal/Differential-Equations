import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(real_2d, pred_2d=None, interval=30):
    fig, ax = plt.subplots()
    ax.set_title("Trajectory Animation")
    ax.set_xlim(real_2d[:, 0].min()-1, real_2d[:, 0].max()+1)
    ax.set_ylim(real_2d[:, 1].min()-1, real_2d[:, 1].max()+1)
    real_line, = ax.plot([], [], lw=2, label="True")
    pred_line, = ax.plot([], [], lw=2, linestyle='--', label="ML Predicted", color='orange')
    ax.legend()

    def update(frame):
        real_line.set_data(real_2d[:frame, 0], real_2d[:frame, 1])
        if pred_2d is not None:
            pred_line.set_data(pred_2d[:frame, 0], pred_2d[:frame, 1])
        return real_line, pred_line

    anim = FuncAnimation(fig, update, frames=len(real_2d), interval=interval, blit=True)
    plt.show()
