# Build the report. `make` regenerates every table and macro from the committed
# results first, so the document cannot drift from the data it describes.
PY ?= ../.venv/bin/python

paper.pdf: paper.tex refs.bib generated/tables.tex generated/numbers.tex
	tectonic -k paper.tex

generated/tables.tex generated/numbers.tex: build_tables.py $(wildcard ../results/*.json)
	$(PY) build_tables.py

figures:
	$(PY) ../experiments/plot_alignment.py
	$(PY) ../experiments/demo_loop.py

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf

.PHONY: figures clean
