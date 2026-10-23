# ChimeraX-Chemur

A UCSF ChimeraX bundle that runs
[Chemur](https://github.com/SweeneyAaron/chemur) interaction prediction on an
open structure and visualizes the predicted interactions as colored pseudobonds, with
both a **command** and a **GUI tool**.

The bundle requires `chemur>=0.1.1,<0.2`, including hydrophobic enclosures, ligand assemblies,
contact projection and the hydrophobic endpoint-hydrogen occlusion exemption. Development
builds can share a version string, so the bundle also checks the required capabilities before
analysis changes atoms or graphics, and reports how to update an incompatible engine.
A repeated `devel install` does not refresh dependencies (see [Install](#install)).

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
  [selectedOnly true|false] [skipProteinNucleic true|false]
  [hydrophobicEnclosure true|false] [assemblyOverrides <file.json>]`
  - `selectedOnly` keeps only interactions with at least one selected partner atom.
  - `skipProteinNucleic` drops protein–protein and nucleic–nucleic contacts (keeps ligand and
    cross-type contacts).
- **Tool:** `Tools ▸ Structure Analysis ▸ Chemur` — pick a structure, set options, click
  *Analyze interactions*, and browse the results **tree**.
  - Primary options: *Add hydrogens*, *Selected atoms only*, *Skip protein–protein /
    nucleic–nucleic*.
  - **Advanced options** (collapsed disclosure): *Protonate ligands*, *Ligand SMILES*,
    *Detect hydrophobic enclosure*, an assembly override JSON file, and the
    per-rule **geometry parameters**, read from the engine’s versioned catalogue and default
    profile. Labels identify the actual distance endpoints and angle definitions; tooltips
    explain chemical scope and link the sources. This includes hydrogen–centroid distances,
    acceptor-side angles and both sulfur–π classifications. Only edited values are overridden;
    *Reset cutoffs to defaults* restores the engine values. The effective profile and geometry
    revision are retained with each run. Cutoff editing is GUI-only — use the `chemur` CLI
    for scripted overrides.
  - Results tree: **Protein / Nucleic acid → chain → residue → interactions** and
    **Ligands / Other → residue → interactions**. A cross-residue interaction is listed under
    *both* endpoints' residues. Each leaf shows distance, angle and offset, with the remaining
    geometry and source metadata in its tooltip. Ligand-centred hydrophobic enclosures have their own
    parent entries, labelled by the enclosed centre, with expandable centre–neighbour
    contact legs. Shared legs are drawn once even when they belong to two enclosures.
  - **Show / hide**: each tree item has a checkbox — untick a leaf to hide one interaction, or a
    residue/chain/section node to hide a whole group (tri-state). A **type legend** above the tree
    (color swatch + count per type) toggles all interactions of a type; *Show all* / *Hide all*
    toggle everything. Visibility drives the ChimeraX pseudobonds directly.
    **Hide intra-chain polymer contacts** applies only to the same live polymer chain in
    the same structure. Ligand–protein and ligand–nucleic acid contacts remain visible even
    when their author chain labels match; matching labels in separate models stay distinct.
  - Clicking a contact selects its two atoms. An enclosure parent selects all mapped
    native participants, including filtered neighbours. A residue/chain/section node
    selects the atoms represented by its children.
    Filters and visibility affect displayed contacts; the complete native assignment is retained.
  - **Export 2D…**: opens a **preview popup** of a 2D *chemical-structure* interaction
    diagram for the selected residues/ligand and their interaction partners. Each unit is drawn as a
    real **RDKit 2D depiction** (whole chemical ligand or receptor residue, without explicit
    hydrogens; aromatic rings, double/triple bonds and **formal charges**), positioned by the real 3D geometry and
    spaced for clarity, with dashed colored bonds between the specific interacting atoms. Save to
    PNG/SVG/PDF from the popup. Select a ligand or residue first.
    The **Colour**, **Black & white** and **Bold** styles share the same chemistry.
    Scroll zoom uses fractional trackpad motion and gentle bounded steps. Per-residue /
    ligand-assembly checkboxes hide complete depictions, labels and their incident contacts;
    enabling a unit restores its previously enabled contacts. Hidden units do not affect
    routing, fit or export bounds. **Font size (pt)** changes all diagram text, including
    chemical atom labels, charges and covalent-boundary anchors (6–24 pt; default 10).
    Restyling preserves unit centres and contact endpoints. Saved images use this same
    visible scene and respect the selected format, appending an omitted extension.
    Native validated templates preserve the complete ligand assembly, including atoms and
    member residues with no displayed contacts. Covalent attachments end in labelled dummy
    sites; updated receptor charges and product bond orders come from engine chemistry.
    Missing or inconsistent ligand chemistry produces an explicitly labelled fallback node.
    Modified polymer products also use their validated templates and diagnostics; unresolved
    products receive labelled fallback nodes. Ordinary residue depictions retain the existing
    ChimeraX Mol2 fallback. Diagram visibility never changes native assigned exports.
- **Trajectory:** `chemur trajectory <structure> [start <n>] [stop <n>] [stride <n>]
  [excludeSolvent true|false] [excludeIons true|false] [hydrophobicEnclosure true|false]
  [assemblyOverrides <file.json>]` analyses loaded coordinate sets. The GUI also provides
  occupancy, per-frame counts, a fingerprint, a frame slider and distance plots.
- **Pose comparison:** native ligand–target contacts populate the comparison and consensus,
  including contacts that cannot be mapped for drawing. Successful analysis with no contacts
  is distinct from a failed analysis; failures show their diagnostic and are excluded from
  consensus denominators. Mapping status separately reports complete, partial or unavailable
  3D mapping. Assigned pose reports can be exported even when drawing is unavailable.

Interactions are drawn in a per-structure pseudobond group named **"Chemur interactions"**,
colored by type (see `src/colors.py`).
Salt bridges connect the charged atoms used for the engine's reported distance.
Amide–π contacts connect the amide C–N midpoint to the ring centre, including
backbone amides spanning two residues. Both residues participate in selection,
and 2D diagrams retain both midpoint atoms when moving or hiding depictions.

### Enclosures, trajectories and counts

An enclosure requires a hydrophobic atom strictly surrounded in three dimensions by
valid hydrophobic neighbours. It uses the engine's ordinary hydrophobic distance cutoff
(4 Å by default), after covalent and occlusion exclusions. Planar and one-sided contacts
do not qualify. Detection is enabled in the default profile; disabling enclosure restores
ordinary hydrophobic contacts. The command preserves a custom profile's setting unless
`hydrophobicEnclosure` is supplied. GUI analyses use the Advanced options checkbox.

One native enclosure is one assigned interaction. Its displayed legs are individual
centre–neighbour contacts, so native interaction totals and drawn-contact totals can differ.
The parent shows the native maximum leg distance; child contacts show their own distances.
Neither filtering nor expanding the tree changes assignment or scoring.

Trajectory enclosure occupancy follows the centre atom even when surrounding partners
change. Clicking an occupancy entry selects the current analysed frame's centre and
partners; if that enclosure is absent, only its centre is selected. Selection-scoped counts
and fingerprint cells use each frame's participants. For enclosures, “both ends selected”
means the centre and at least one surrounding atom, rather than every surrounding residue.
Global occupancy and native trajectory exports retain the full analysed results.

### Exported data

- **Export assigned results…** writes native Chemur JSON or CSV from the active result.
  It includes hidden and unmapped assigned interactions and preserves grouped enclosure
  records with centre, neighbours and leg metadata. No raw candidates are requested or exported.
- **Export displayed contacts…** writes the visible pair contacts in the chosen scope:
  all displayed contacts, selected tree rows or contacts touching selected atoms. Selecting
  an enclosure parent includes its displayed children; selecting a child includes that leg.
  Shared edges appear once with all enclosure provenance. Existing partner and geometry
  columns are retained, with complete metadata encoded as JSON in CSV and as objects in JSON.
- **Export assigned trajectory…** uses the engine's trajectory serializer, retaining frame
  indices, per-frame metadata and, in JSON, occupancy. Pose export writes one native JSON/CSV
  pair for each successful pose analysis.

Choose JSON or CSV in the save dialog. A missing extension is added; an extension that
conflicts with the selected format is rejected rather than silently changing the format.
The separate raw-report option in Chemur's standalone CLI does not alter these plugin exports.

### Chemical corrections and model ownership

SMILES corrections accept residue names (`LIG:CCO` or `LIG=CCO`) or a full component identity
(`A:LIG:401=CCO`). The ligand report's inline corrections use component identity, so correcting
one copy need not change every ligand with the same residue name. Related ligand residues
are reported as one chemical assembly. For connected or covalently attached products, use
**Load assembly override JSON…** or the command's `assemblyOverrides` argument with the
engine's validated product-chemistry schema. Corrections are associated with their source
model and are cleared when that model closes.

The results label identifies the model and analysis mode that own the visible result.
Changing the input dropdown does not relabel an older result as a new analysis. Closing a
source model invalidates only analyses that depend on it and closes their owned graphics,
plots and dialogs. Closing one pose removes that pose from its comparison; closing its
receptor invalidates the comparison. Moving a pose relative to the receptor requires
reanalysis. Analysis cannot be started again while another run is active, and late callbacks
cannot restore results after their source or tool has closed.

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
# 1. Install or upgrade to a compatible engine, through ChimeraX's OWN pip.
#    Not "<chimerax>/bin/python3.11 -m pip" -- that can land in your user
#    site-packages, which ChimeraX does not load.
"$CX" --nogui --exit --cmd 'pip install "chemur>=0.1.1,<0.2" upgrade true'

# 2. Check the engine location, capabilities and selected geometry backend.
#    Both C++ and Python backends are supported; True is not required.
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

For local development, install the matching wheel built from the current Chemur source
in ChimeraX's environment. Older `0.1.0` builds no longer satisfy the dependency requirement.
Verify the loaded path and capability report, not just `pip show`'s
version string. Source-based tests below can validate the plugin against a working engine
checkout before packaging it.

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
pip install "chemur>=0.1.1,<0.2"
```

`USING_CPP_CORE = False` is supported: the engine uses its Python geometry backend when
the compiled extension is unavailable or `CHEMUR_FORCE_PYTHON_CORE=1` is set. The C++ backend
is generally faster. Both backends must pass the same interaction tests; an old compiled
extension can use the engine's Python fallback for newer geometry operations. Backend
selection and engine API compatibility are separate checks.

## Tests

The pure-logic parts of the bundle (exports, SDF preparation, native diagram chemistry,
pose comparison, result ownership and engine→ChimeraX mapping) are unit-tested without ChimeraX. The tests load
modules by file path and stub the `chimerax.*` imports, so they run in an **ordinary venv or
conda env — not ChimeraX's Python**:

Use an environment with the updated engine's runtime dependencies installed, then run
from the plugin repository:

```bash
pip install pytest rdkit
# Point to the updated engine checkout, not an older installed wheel.
ENGINE_SRC=/absolute/path/to/chemur/src
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ENGINE_SRC" python -m pytest -p no:cacheprovider
CHEMUR_FORCE_PYTHON_CORE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ENGINE_SRC" python -m pytest -p no:cacheprovider
```

`tests/test_interaction_names.py` additionally checks that `src/colors.py`'s style table
still covers exactly the interaction types the installed engine emits — drift there is
otherwise silent (an unstyled type just draws grey and unlabelled).

The suite covers enclosure projection and shared legs, assigned-versus-display exports,
source-model invalidation, failed-pose consensus, partial mapping, changing trajectory
shells, scene coordinates, typed hydrogen orientation and validated assembly diagrams.
Saved real enclosure results under `tests/fixtures/1MHT_enclosure.json` and
`tests/fixtures/4W53_enclosure.json` allow deterministic checks without network downloads.
Headless tests use RDKit for chemistry but may stub raster rendering; they do not replace
an installed ChimeraX smoke test for Qt, pseudobonds, markers, interactive diagrams and
model-close triggers.

Run the headless checks inside ChimeraX from this repository. `--engine-source` is optional
when its installed engine already has the required capabilities:

```bash
"$CX" --nogui --exit --cmd "runscript tools/engine_check.py --engine-source /absolute/path/to/chemur/src"
"$CX" --nogui --exit --cmd "runscript tools/smoke_test.py /absolute/path/to/4W53.cif /absolute/path/to/1MHT.cif --engine-source /absolute/path/to/chemur/src --output-dir /tmp/chemur-smoke"
```

The engine check prints the loaded location, backend and individual capability checks.
The smoke test loads this plugin's source modules, analyses the two structures, verifies
native enclosure groups and displayed legs, writes both export formats, checks mapping
after a rigid scene transform and verifies cleanup when the source closes. Success ends
with `CHEMUR SMOKE PASS`. Input structures are local mmCIF files; the saved JSON fixtures
are for unit tests and do not replace those coordinate files. To repeat with Python
geometry, prefix the ChimeraX command with `CHEMUR_FORCE_PYTHON_CORE=1`.

The smoke harness uses the ordinary analysis path with no chemistry overrides. Both
structures are checked without hydrogen addition; 1MHT is also reopened and checked with
hydrogen addition enabled. SAH:C4 retains its four enclosure legs in both cases, and heavy
atom coordinates remain unchanged. Named ligand chemistry is retained from the source
mmCIF during export. Inline modified `C36` nucleotides remain unresolved in this
no-override harness; explicitly loading the engine’s reviewed per-instance
`examples/assembly_overrides/1MHT-C36-products.json` validates their linked chemical products.

The placement check tests independently transformed receptor and ligand models through
both mmCIF and SDF, including hydrogen preparation, local ring-marker coordinates and
cleanup:

```bash
"$CX" --nogui --notools --exit --cmd "runscript tools/placement_smoke_test.py /absolute/path/to/4W53.cif --engine-source /absolute/path/to/chemur/src"
```

Run the GUI smoke test in a separate ChimeraX GUI session, without `--nogui`:

```bash
"$CX" --notools --exit --cmd "runscript tools/gui_smoke_test.py /absolute/path/to/4W53.cif --engine-source /absolute/path/to/chemur/src"
```

This accepts either 4W53.cif or 1MHT.cif without chemistry overrides and exercises
actual Qt enclosure children, tri-state visibility, participant selection,
interactive diagrams, trajectory plots, static/trajectory ownership, native and displayed
exports, preservation of the previous view after a failed run,
closing and reopening a model with a reused ID, and closing the source or deleting the
tool during progress callbacks. The test opens and closes its own models and tool; use
the separate session so it does not interrupt an interactive analysis.
Append `--add-hydrogens` to the script arguments to exercise the GUI's **Add hydrogens**
option; use 1MHT.cif to check that SAH:C4 still has its four enclosure contacts.

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
  open structure. Set `addHydrogens false` to skip. For hydrophobic contacts, including
  enclosure legs, hydrogens directly bonded to either endpoint are exempt from occlusion.
  Unrelated hydrogens and heavy atoms still undergo the existing occlusion checks, and
  other interaction types are unchanged. In 1MHT, adding PHE18:HD1 therefore preserves
  SAH:C4's four surrounding hydrophobic contacts and its enclosure.
  Turning the option off does not remove hydrogens added by an earlier run.
- **New hydroxyl hydrogens can be oriented automatically.** RDKit's coordinates for a
  newly generated hydroxyl or thiol hydrogen can have an arbitrary torsion, affecting
  which hydrogen bonds the engine detects. Existing experimental or user-provided
  hydrogens are preserved.
  - For a **whole structure** (`chemur interactions`, or a pose that already contains the
    protein), ChimeraX's own `addh` does an H-bond-aware search and places them well.
  - For a **ligand-only SDF pose** analysed against a receptor — the *Compare docked poses*
    route — `addh` is deliberately not run on the ligand, because it can break the SDF's valence
    so RDKit then fails to sanitise it. RDKit adds the hydrogens instead, and RDKit cannot see
    the receptor, so the bundle orients the rotatable polar hydrogens against the receptor's
    nearby chemically typed acceptors itself (`sdf_io.orient_polar_hydrogens`). Receptor
    acceptors are prepared from the same engine chemistry after model preparation;
    intraligand acceptors use the engine's RDKit annotations. Amide and cationic nitrogens
    are not treated as acceptors just because they are nitrogen, and covalent 1–2/1–3
    neighbours are excluded. Only newly generated hydrogens can rotate; every heavy atom
    and every pre-existing hydrogen remains at its scene coordinate. If receptor chemistry
    cannot be prepared safely, optional orientation is skipped.
  - Conjugated hydroxyls (phenols, carboxylic acids, enols) are restricted to their two in-plane
    torsions, since swinging one out of plane to manufacture a hydrogen bond would be wrong.
  - This preparation applies to ligand-only pose analysis against a receptor. It is skipped
    when `addHydrogens` is false, because that run does not generate new hydrogens.
- Ligand chemistry: the engine uses available embedded chemistry, explicit corrections and
  validated assembly templates, with CCD lookup when needed. The temporary mmCIF retains
  named component atom and bond dictionaries from model metadata or the original local
  mmCIF, after checking compatibility with the current heavy atoms and bonds. Coordinates
  and inter-residue connectivity come from the current models; incompatible source dictionaries
  are skipped with a warning. Network access may be required
  for missing templates. `ligandSmiles` can target a residue name or a full component ID;
  use assembly override JSON for final product chemistry across connected components.
- The engine is invoked by saving the chosen model to a temporary mmCIF file and calling
  `chemur.analyze()`; ligand-only SDF poses preserve their bond orders through the SDF route.
  Coordinates are aligned in the ChimeraX scene frame. Back-mapping combines source identity
  with coordinate agreement, so identically named atoms in different copies do not become
  arbitrary endpoints. Missing or ambiguous mappings remain unavailable for drawing while
  the native result stays exportable.
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


### Modified polymer chemistry and engine definitions

The plugin requires the matching engine with the versioned geometry catalogue and modified
polymer preparation API; an older engine is rejected before analysis changes a structure.
Advanced options and `tools/engine_check.py` report the engine path, geometry backend and
geometry-definition revision. Python and C++ geometry backends are both supported.

Live mmCIF exchange preserves deposited polymer membership and sequence positions, named
component chemistry and per-residue source entry identity. It retains current scene coordinates
and live bonds. Separate models with duplicate complete component identities must receive
distinct chain identifiers before joint analysis, because the engine cannot distinguish those
components safely. Modified CCD monomers remain polymer residues, including inline C36 in
1MHT, instead of being treated as standalone ligands.

The **Chemical product report…** includes ligand assemblies and modified polymer products,
with validated product SMILES, charge and diagnostic messages. Modified polymer and attached
products are read-only and accept complete assembly-override JSON documents. For 1MHT the
engine provides reviewed source-scoped examples under `examples/assembly_overrides/`; load
them explicitly using **Load assembly override JSON…** or `assemblyOverrides`. Corrections
are not inferred from a residue name or applied automatically to unrelated entries.

`anion_pi` is no longer an active detector. Historical results retain their display style.
Raw candidate collection/output remains disabled, and native assigned exports remain independent
of mapping failures and display filters.
