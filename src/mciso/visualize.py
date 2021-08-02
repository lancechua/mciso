import matplotlib.pyplot as plt
import pandas as pd


def scenarios_by_product(
    X: "np.ndarray", indices: list, products: list, ax: plt.Axes = None
) -> plt.Axes:
    """Plot generated scenarios, with a subplot for each product"""
    if ax is None:
        _, ax = plt.subplots(X.shape[-1], 1, figsize=(8, X.shape[-1] * 2), sharex=True)

    try:
        iter(ax)
    except TypeError:
        ax = [ax]

    for i, prod_i in enumerate(products):
        pd.DataFrame(
            X[:, :, i],
            index=indices,
        ).plot(ax=ax[i], alpha=0.05, linewidth=3, legend=None, color="gray")

        pd.DataFrame(X[:, :, i].mean(axis=1), index=indices, columns=["avg"]).plot(
            ax=ax[i], alpha=0.8, linewidth=1, legend=None, color="blue"
        )

        ax[i].set_ylabel(prod_i)

    return ax
