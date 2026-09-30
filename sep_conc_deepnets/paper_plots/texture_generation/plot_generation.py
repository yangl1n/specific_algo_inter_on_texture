"""Plot example / synthesis pairs in a compact one-row figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "texture_generation.png"

PAIRS = [
    ("14.jpg", "14_synth.png"),
    ("20.jpg", "20_synth.png"),
]


def main():
    n_pairs = len(PAIRS)
    # spacer column between pairs, slightly wider than the intra-pair gap
    width_ratios = []
    for i in range(n_pairs):
        if i:
            width_ratios.append(0.12)
        width_ratios.extend([1, 1])

    fig = plt.figure(figsize=(6.75, 1.95), dpi=200)
    gs = GridSpec(
        1,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.04,
        left=0.005,
        right=0.995,
        top=0.84,
        bottom=0.02,
    )

    col = 0
    for example_name, synth_name in PAIRS:
        for title, fname in (("Example", example_name), ("Synthesis", synth_name)):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(Image.open(HERE / fname).convert("RGB"))
            ax.set_title(title, fontsize=9, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            col += 1
        col += 1  # skip spacer

    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
