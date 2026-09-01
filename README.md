# ChimeraX-Chemur

A UCSF ChimeraX bundle that runs
[Chemur](https://github.com/SweeneyAaron/chemur) interaction prediction on an
open structure and visualizes the predicted interactions as colored pseudobonds, with
both a **command** and a **GUI tool**.

The bundle declares `chemur` as a dependency, so a normal install pulls the engine from
PyPI automatically — but a repeated `devel install` does not (see [Install](#install)).

> **Renamed from ChimeraX-ChemeleonX.** The bundle, the ChimeraX module, both commands,
> the GUI tool and the underlying engine all changed name. pip treats `ChimeraX-Chemur`
> as a brand-new distribution, so it will **not** replace `ChimeraX-ChemeleonX` in place —
> both would stay installed and both would register commands. Run these in ChimeraX's
> command line, then restart ChimeraX:
>
> ```
> toolshed uninstall ChimeraX-ChemeleonX
> pip uninstall chemeleonx
> ```
>
> Then install this bundle — see [Install](#install) below.
>
> If you go all the way back to the original release, remove that pair too:
> `toolshed uninstall ChimeraX-ProPLID` and `pip uninstall proplid`.
>
> The two uninstalls use different commands on purpose: ChimeraX's `pip` refuses any
> package name starting with "chimerax", so the bundle must go through
> `toolshed uninstall`; and `toolshed uninstall` does not remove dependencies, so the
> engine must go through `pip uninstall`.
>
> | Old | New |
> |---|---|
> | `chemeleonx interactions` | `chemur interactions` |
> | `chemeleonx trajectory` | `chemur trajectory` |
> | `Tools ▸ Structure Analysis ▸ ChemeleonX` | `Tools ▸ Structure Analysis ▸ Chemur` |
>
> Sessions saved with the old bundle restore groups named "ChemeleonX interactions" /
> "ChemeleonX π-centres" / "ChemeleonX pose …". The new bundle draws into "Chemur …"
> groups and will not clear the old ones, so you may see doubled interaction lines.
> Close the leftover models once from the Model Panel — note that `delete H` will *not*
> remove the stale ring-centre markers, which are deliberately elemented as He.

## What it provides

- **Command:** `chemur interactions <structure> [protonate true|false]
  [profile <name>] [ligandSmiles "LIG:CCO,ABC:c1ccccc1"] [addHydrogens true|false]
  [selectedOnly true|false] [skipProteinNucleic true|false]`
  - `selectedOnly` keeps only interactions with at least one selected partner atom.
  - `skipProteinNucleic` drops protein–protein and nucleic–nucleic contacts (keeps ligand and
    cross-type contacts).
- **Tool:** `Tools ▸ Structure Analysis ▸ Chemur` — pick a structure, set options, click
  *Analyze interactions*, and browse the results **tree**.
  - Primary options: *Add hydrogens*, *Selected atoms only*, *Skip protein–protein /
    nucleic–nucleic*.
  - **Advanced options** (collapsed disclosure): *Protonate ligands*, *Ligand SMILES*, and the
    per-rule **geometry cutoffs** (distance/angle/offset/donor-angle, seeded from the default
    profile; any value you change from its default is applied; *Reset cutoffs to defaults*
    restores them). Cutoff editing is GUI-only — use the `chemur` CLI for scripted overrides.
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
  - **Export 2D…**: opens a **preview popup** of a 2D *chemical-structure* interaction
    diagram for the selected residues/ligand and their interaction partners. Each unit is drawn as a
    real **RDKit 2D depiction** (whole residue, no hydrogens; aromatic rings, double/triple bonds,
    and **formal charges** — Asp/Glu −1, Lys/Arg +1, etc.), positioned by the real 3D geometry and
    spaced for clarity, with dashed colored bonds between the specific interacting atoms. Save to
    PNG/SVG/PDF from the popup. With nothing selected it uses all shown interactions.
    Bond orders come from ChimeraX (Mol2); formal charges use a standard residue table (plus
    local functional-group detection for ligands); a unit that can't be depicted degrades to a
    labelled node.

Interactions are drawn in a per-structure pseudobond group named **"Chemur interactions"**,
colored by type (see `src/colors.py`).

## Install

> **Not on the ChimeraX Toolshed yet.** `toolshed install ChimeraX-Chemur` does not work
> today — build and install from this repo instead (below). Once the bundle is published,
> installing from `Tools ▸ More Tools…` will pull the `chemur` engine from PyPI as a
> dependency, with no compiler needed: prebuilt wheels cover ChimeraX's CPython 3.11 on
> macOS, Linux and Windows.

### Build and install from source

Everything here runs **inside ChimeraX**, through its own Python. The PEP 517 backend this
bundle declares, `ChimeraX-BundleBuilder`, ships through the ChimeraX Toolshed and is **not
on PyPI** — so `pip install .` and `python -m build` from an ordinary venv cannot work, and
will fail while trying to fetch the backend. Use ChimeraX's `devel` command instead.

Set `CX` to your launcher. ChimeraX 1.8 / 1.9 / 1.10 all use Python 3.11; adjust the
version in the path as needed.

```bash
# macOS
CX="/Applications/ChimeraX-1.10.1.app/Contents/bin/ChimeraX"

# Linux -- distro package, or the tarball/installer location
CX="chimerax"                                  # or: /opt/UCSF/ChimeraX/bin/ChimeraX
```

Then, **from the root of this repo**:

```bash
# 1. Install the engine, through ChimeraX's OWN pip.
#    Not "<chimerax>/bin/python3.11 -m pip" -- that can land in your user
#    site-packages, which ChimeraX does not load.
"$CX" --nogui --exit --cmd "pip install chemur"

# 2. Confirm it landed and that the compiled core is active (must print True).
#    There is no `shell` COMMAND in ChimeraX (Shell is a Tool), so use `runscript`.
"$CX" --nogui --exit --cmd "pip show chemur"
"$CX" --nogui --exit --cmd "runscript tools/engine_check.py"

# 3a. Build a wheel only -- it lands in dist/.
"$CX" --nogui --exit --cmd "devel build . exit true"

# 3b. Or build AND install into ChimeraX -- normally what you want.
"$CX" --nogui --exit --cmd "devel install . exit true"
```

**Run `devel` from this directory.** Both `devel build` and `devel install` resolve the
bundle's `src/` relative to the *current working directory*, not the path argument — run
them from anywhere else and they will not find the source.

Re-run step 3b (from this directory) after each change to the bundle source, then restart
ChimeraX.

**Step 1 is not optional, and `devel install` will not do it for you.** When the bundle
version you are building already matches the installed one, `devel install` appends
`reinstall true`, and ChimeraX's installer then adds `--force-reinstall --no-deps` — so
`chemur` and `rdkit` are silently skipped on every dev iteration after the first.
`devel install . noDeps false` does not help; the `--no-deps` is added unconditionally in
the reinstall branch. Note this is invisible on day one: the *first* install of a new
bundle name is not a reinstall, so dependencies do get pulled that once.

To force a same-version engine refresh (e.g. iterating on a local engine build): ChimeraX's
`pip` command has **no** `--force-reinstall`. It takes one package requirement plus
`upgrade`/`verbose`, and validates the requirement, so `pip install --force-reinstall chemur`
fails with "invalid requirement specified". Use two steps:

```
pip uninstall chemur
pip install chemur
```

`USING_CPP_CORE = False` is not an error — the engine silently degrades to a much slower
pure-Python core when `_chemur_core` is unavailable. Prebuilt cp311 wheels exist for macOS
universal2, manylinux_2_28 x86-64 and win_amd64, so `False` on those platforms means the
sdist fallback ran and something went wrong.

## Tests

The pure-logic parts of the bundle (`export`, `sdf_io`, `pose_compare`, and the
engine→ChimeraX atom mapping in `runner`) are unit-tested without ChimeraX. The tests load
modules by file path and stub the `chimerax.*` imports, so they run in an **ordinary venv or
conda env — not ChimeraX's Python**:

```bash
pip install pytest rdkit chemur
pytest
```

`tests/test_interaction_names.py` additionally checks that `src/colors.py`'s style table
still covers exactly the interaction types the installed engine emits — drift there is
otherwise silent (an unstyled type just draws grey and unlabelled).

## Try it

In ChimeraX:

```
open 3poz
chemur interactions #1
```

Pseudobonds appear and a per-type summary is logged. Or open the GUI from
`Tools ▸ Structure Analysis ▸ Chemur`.

## Notes

- `addHydrogens` defaults to **true**: Chemur needs explicit donor hydrogens to assign
  H-bond-like interactions, so the bundle runs `addh` on the model first. This modifies the
  open structure. Set `addHydrogens false` to skip.
- Ligand chemistry: by default Chemur fetches SMILES for known ligands from the CCD (needs
  network). Provide `ligandSmiles` to override per residue name.
- The engine is invoked by saving the chosen model to a temporary mmCIF file and calling
  `chemur.analyze()`; results are mapped back onto the live ChimeraX atoms by
  `(chain, residue, atom name)`.
- **Panel width.** The tool docks at ~340 px and every row reflows to fit — button rows wrap
  onto extra lines rather than scrolling sideways, and long text (file paths, model names,
  section intros) is shortened on screen with the full string on hover. Drag the dock edge to
  resize; it goes down to 180 px.
  If you have ever used *Save Tool Position* on this panel, that saved width wins over the
  default. To change it, drag the panel to the width you want and right-click its title bar ▸
  **Save Tool Position** again — there is no built-in "forget", but you can clear it from
  ChimeraX's Python shell:

  ```python
  p = session.ui.settings.tool_positions
  p["windows"].pop("Chemur", None)
  session.ui.settings.tool_positions = p   # reassign: the setter is what persists
  session.ui.settings.save()
  ```

## License

MIT — see [LICENSE](LICENSE).
