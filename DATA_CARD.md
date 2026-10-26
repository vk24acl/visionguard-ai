# Dataset provenance and scope

Source: scikit-learn `load_digits`, a bundled copy of the UCI optical recognition
digits test subset. 1,797 examples, 10 classes, 64 integer features in [0,16].
Each image is 8×8. No external download occurs when training this project.

- Official loader: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- Original source: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Citation: E. Alpaydin and C. Kaynak, Optical Recognition of Handwritten Digits,
  UCI Machine Learning Repository (1998), DOI: 10.24432/C50P49.

UCI lists this dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: E. Alpaydin and C. Kaynak (1998), linked above. Demo pixels are rescaled from [0,16] to [0,1]; interactive transformations are described below.

The dataset has its own source terms, separate from the MIT licence on this
project's code. This repository includes only 40 transformed held-out images
for the interactive demo, plus trained weights and aggregate metrics. It does
not redistribute the full dataset. Check upstream terms for further reuse.

Stratified random partitions are fixed with seeds 42, 43 and 44. Writer IDs are
not exposed by the bundled loader, so partitions are not writer-independent.
Noise σ=.25, middle-row occlusion and zero-padded right shifting are artificial
stress tests. They are not representative of every real scanning condition.
