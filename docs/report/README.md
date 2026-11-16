# Build the technical report

`main.tex` builds `RAG_Phinha.pdf` with the IEEEtran conference class,
A4 paper and two columns.

The report includes three vector figures: the inference pipeline, a synthetic
training-reference example and the external score differences. The example uses
invented labels. The score figure uses external development means only.

Use an existing TeX Live or MiKTeX installation with IEEEtran, amsmath, amssymb,
graphicx, booktabs, cite, url and hyperref. From this directory:

```bash
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

On Windows, create `build` with `New-Item -ItemType Directory -Force build`
and run the same two `pdflatex` commands. Check each exit code before continuing.
The source includes the bibliography. Compile with shell escape disabled.
Review `build/main.pdf`, then copy it to `RAG_Phinha.pdf` in this directory.
The repository and delivery package must use the same PDF bytes.
Keep auxiliary files in `build/`; they are not source files.

The included vector PDFs in `figures/` are ready for compilation.
To edit or regenerate them, use an existing Python environment with Matplotlib
and NumPy. From this directory, run:

```bash
python make_figures.py
```

`external_metrics.json` contains the audited external aggregates. The generator
writes the three figure PDFs and `figure_data.json`, which records the nine
plotted differences and the source hash.
Recompile the report after changing a figure.

The metrics JSON also includes the 21 external match comparisons behind the
seven-of-seven claim for the two-second reference, with their original source
commit and file hash. No challenge frames or predictions are included.

Template: [IEEE Author Center](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/),
[IEEEtran 1.8b](https://ctan.org/pkg/ieeetran), LaTeX Project Public License 1.3.
Reproducibility and attribution follow the applicable parts of the
[CVPR suggested practices](https://cvpr.thecvf.com/Conferences/2025/AuthorSuggestedPractices).
Dataset versions and development history are recorded in
[SOURCES](../SOURCES.md) and [METHOD](../METHOD.md).
