import time
import seaborn as sns
import colorcet as cc


def set_plot_style():
    sns.set_context("paper")
    sns.set(font='serif')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"],
        'legend.frameon': False
    })


COLOR_PALETTE = sns.color_palette(cc.glasbey_dark, 256)


def timer(func):
    """Decorator that prints the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timer_end_message = f"  :):):) Finished {func.__name__!r} in {run_time:.4f} seconds"
        if "case" in kwargs:
            timer_end_message += f" / SCENARIO={kwargs['case']}"
        if "seed" in kwargs:
            timer_end_message += f" / SEED={kwargs['seed']}"
        if "path" in kwargs:
            timer_end_message += f" / Results stored in {kwargs['path']}"
        print(timer_end_message)
        return value

    return wrapper_timer
