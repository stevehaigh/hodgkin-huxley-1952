# Reconstructing Hodgkin and Huxley (1952)

A computational reconstruction of Hodgkin, A. L. & Huxley, A. F. (1952),
"A quantitative description of membrane current and its application to
conduction and excitation in nerve", *Journal of Physiology* **117**, 500–544
([doi](https://doi.org/10.1113/jphysiol.1952.sp004764) ·
[page scans on PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC1392413/)).

Every number here comes from the constants printed in that paper. Nothing is
refitted or retuned, and the paper's own sign convention is kept throughout,
so depolarisation is negative and $V_{Na} = -115$ mV.

The notebook walks the deduction in order: fit six empirical functions to
voltage-clamp records, assemble them into one differential equation, and then
find out what the equation knows that nobody put into it.

```
notebooks/hodgkin-huxley-1952.ipynb
```

## What comes out

| | reconstruction | paper |
|---|---|---|
| resting $m$, $h$, $n$ | 0.0529, 0.5961, 0.3177 | p. 520 |
| peak of the 15 mV spike | 105.4 mV at 1.16 ms | ≈1.1 ms, Fig. 12 |
| firing threshold at 6.3 °C | 6.50 mV | "between 6 and 7 mV" |
| $K$ in eqn (31) | 10.458 msec⁻¹ | 10.47, p. 528 |
| conduction velocity | 18.75 m/sec | 18.8 calculated, 21.2 measured |

The conduction velocity is the one that matters. The equations were fitted to
a membrane clamped at fixed voltages, then asked how fast a wave would travel
down an axon, which is a question about a completely different experiment.
They came back within 12%.

Threshold, the refractory period, and anode break excitation all fall out of
the same equations without having been fitted, mentioned, or encoded anywhere.

## Running it

```sh
make install     # pip install -e ".[notebook,dev]"
make test        # 24 regression tests against the paper's published numbers
make notebook    # regenerate and execute; takes about 15 seconds
make figures     # render every figure to figures/
```

The notebook is generated from `build_notebook.py` rather than edited in
place, so the prose and equations stay diffable as ordinary Python strings.
Edit that file, not the `.ipynb`.

## Layout

```
hh1952/model.py       the equations and the constants, nothing else
hh1952/figures.py     figure builders, all computed from model.py
build_notebook.py     the notebook's prose and structure
tests/test_model.py   regression tests pinning the paper's numbers
```

`model.py` offers two integrators everywhere. `method="lsoda"` is the default;
`method="euler"` is a fixed-step loop standing in for the hand computation of
p. 523. They agree on every published number to four significant figures,
which the notebook checks.

## On copyright

This repository does not reproduce the paper, and cannot. Hodgkin & Huxley
(1952) is under copyright until 2047 in the United States and the end of 2082
in the United Kingdom, held by The Physiological Society. PMC hosts the page
scans free to read, and is [explicit](https://pmc.ncbi.nlm.nih.gov/about/copyright/)
that free access is not a licence to redistribute.

So: equations are set in full, because equations are facts. Short passages are
quoted with page numbers, for criticism and review. No scanned artwork appears
anywhere, and neither do any experimental data. Where H&H drew a curve through
measured points, only the curve is drawn here, and the measured points are
left where they belong.

For the paper verbatim with commentary, the licensed edition is Raman &
Ferster, *The Annotated Hodgkin and Huxley: A Reader's Guide* (Princeton,
2021).

The code and prose in this repository are MIT licensed. See `LICENSE`.
