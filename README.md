# ChimeraX-ChemeleonX

A UCSF ChimeraX bundle that runs
[ChemeleonX](https://github.com/SweeneyAaron/chemeleonx) interaction prediction on an
open structure and visualizes the predicted interactions as colored pseudobonds, with
both a **command** and a **GUI tool**.

The bundle declares `chemeleonx` as a dependency, so installing it pulls the engine
from PyPI automatically.

> **Renamed from ChimeraX-ProPLID.** pip treats this as a different distribution, so
> it will not upgrade an existing install in place. Run
> `toolshed uninstall ChimeraX-ProPLID` (and remove any `proplid` engine copies) first.
> The commands are now `chemeleonx interactions` / `chemeleonx trajectory`.

## What it provides

- **Command:** `chemeleonx interactions <structure> [protonate true|false]
  [profile <name>] [ligandSmiles "LIG:CCO,ABC:c1ccccc1"] [addHydrogens true|false]
  [selectedOnly true|false] [skipProteinNucleic true|false]`
  - `selectedOnly` keeps only interactions with at least one selected partner atom.
  - `skipProteinNucleic` drops protein–protein and nucleic–nucleic contacts (keeps ligand and
    cross-type contacts).
- **Tool:** `Tools ▸ Structure Analysis ▸ ChemeleonX` — pick a structure, set options, click
  *Analyze interactions*, and browse the results **tree**.
  - Primary options: *Add hydrogens*, *Selected atoms only*, *Skip protein–protein /
    nucleic–nucleic*.
  - **Advanced options** (collapsed disclosure): *Protonate ligands*, *Ligand SMILES*, and the
    per-rule **geometry cutoffs** (distance/angle/offset/donor-angle, seeded from the default
    profile; any value you change from its default is applied; *Reset cutoffs to defaults*
    restores them). Cutoff editing is GUI-only — use the `chemeleonx` CLI for scripted overrides.
  - Results tree: **Protein / Nucleic acid → chain → residue → interactions** and
    **Ligands / Other → residue → interactions**. A cross-residue interaction is listed under
    *both* endpoints' residues. Each leaf shows distance, angle and offset, with the remaining
    geometry (donor angle, point–plane angle, geometry, occlusion) in its tooltip.
  - **Show / hide**: each tree item has a checkbox — untick a leaf to hide one interaction, or a
    residue/chain/section node to hide a whole group (tri-state). A **type legend** above the tree
    (color swatch + count per type) toggles all interactions of a type; *Show all* / *Hide all*
    toggle everything. Visibility drives the ChimeraX pseudobonds directly.
  - Clicking a leaf selects its two atoms; clicking a residue/chain/section node selects all
    atoms of the interactions beneath it.
  - **Export 2D diagram…**: opens a **preview popup** of a 2D *chemical-structure* interaction
    diagram for the selected residues/ligand and their interaction partners. Each unit is drawn as a
    real **RDKit 2D depiction** (whole residue, no hydrogens; aromatic rings, double/triple bonds,
    and **formal charges** — Asp/Glu −1, Lys/Arg +1, etc.), positioned by the real 3D geometry and
    spaced for clarity, with dashed colored bonds between the specific interacting atoms. Save to
    PNG/SVG/PDF from the popup. With nothing selected it uses all shown interactions.
    Bond orders come from ChimeraX (Mol2); formal charges use a standard residue table (plus
    local functional-group detection for ligands); a unit that can't be depicted degrades to a
    labelled node.

Interactions are drawn in a per-structure pseudobond group named **"ChemeleonX interactions"**,
colored by type (see `src/colors.py`).

## Install

From the ChimeraX Toolshed (`Tools ▸ More Tools…`), or in ChimeraX's command line:

```
toolshed install ChimeraX-ChemeleonX
```

That pulls the `chemeleonx` engine from PyPI as a dependency — no compiler needed;
prebuilt wheels cover ChimeraX's CPython 3.11 on macOS, Linux and Windows.

## Local install (development)

These steps target ChimeraX 1.10.1 on macOS; adjust the app path for 1.8 / 1.9 (all use
Python 3.11). `CX` below is the ChimeraX app bundle.

```bash
CX="/Applications/ChimeraX-1.10.1.app"

# 1. Install the engine, through ChimeraX's OWN pip.
#    Not "$CX/Contents/bin/python3.11" -m pip -- that can land in your macOS user
#    site-packages, which ChimeraX does not load.
"$CX/Contents/bin/ChimeraX" --nogui --exit --cmd "pip install chemeleonx"

# 2. Confirm the compiled core loaded (must print True):
"$CX/Contents/bin/ChimeraX" --nogui --exit \
  --cmd "shell python -c 'from chemeleonx.core import USING_CPP_CORE; print(USING_CPP_CORE)'"

# 3. Build + install this bundle into ChimeraX.
#    IMPORTANT: `devel install` resolves the bundle's `src/` relative to the *current
#    working directory*, not the path argument. Run it from THIS directory.
cd <this repo>
"$CX/Contents/bin/ChimeraX" --nogui --exit --cmd "devel install . exit true"
```

Re-run step 3 (from this directory) after each change to the bundle source, then restart
ChimeraX. Step 1 only needs re-running when the engine itself changes — and note that a
same-version reinstall is a silent no-op under ChimeraX's `--upgrade-strategy
only-if-needed`, so use `pip install --force-reinstall chemeleonx` when iterating on a
local engine build.

## Tests

The pure-logic parts of the bundle (`export`, `sdf_io`, `pose_compare`, and the
engine→ChimeraX atom mapping in `runner`) are unit-tested without ChimeraX:

```bash
pip install pytest rdkit chemeleonx
pytest
```

`tests/test_interaction_names.py` additionally checks that `src/colors.py`'s style table
still covers exactly the interaction types the installed engine emits — drift there is
otherwise silent (an unstyled type just draws grey and unlabelled).

## Try it

In ChimeraX:

```
open 3poz
chemeleonx interactions #1
```

Pseudobonds appear and a per-type summary is logged. Or open the GUI from
`Tools ▸ Structure Analysis ▸ ChemeleonX`.

## Notes

- `addHydrogens` defaults to **true**: ChemeleonX needs explicit donor hydrogens to assign
  H-bond-like interactions, so the bundle runs `addh` on the model first. This modifies the
  open structure. Set `addHydrogens false` to skip.
- Ligand chemistry: by default ChemeleonX fetches SMILES for known ligands from the CCD (needs
  network). Provide `ligandSmiles` to override per residue name.
- The engine is invoked by saving the chosen model to a temporary mmCIF file and calling
  `chemeleonx.analyze()`; results are mapped back onto the live ChimeraX atoms by
  `(chain, residue, atom name)`.

## License

MIT — see [LICENSE](LICENSE).
