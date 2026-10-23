"""Carry deposited named chemistry through ChimeraX's coordinate-only export.

ChimeraX's mmCIF writer does not write component atom or bond dictionaries.
Recover those dictionaries without copying deposited coordinates or connections
over the current structure.  No ChimeraX imports or model mutations are needed.
"""
from collections import defaultdict
from pathlib import Path


_CATEGORIES = ("_chem_comp_atom.", "_chem_comp_bond.",
               "_pdbx_chem_comp_descriptor.", "_chem_comp.")
_HYDROGENS = {"H", "D", "T"}
_SOURCE_CATEGORIES = _CATEGORIES + (
    "_entry.", "_atom_site.", "_entity.", "_entity_poly.", "_entity_poly_seq.",
    "_pdbx_poly_seq_scheme.", "_pdbx_struct_mod_residue.")


def _rows(block, category):
    columns = block.get_mmcif_category(category)
    return [dict(zip(columns, values)) for values in zip(*columns.values())] if columns else []


def _present(value):
    return value not in (None, False, "", ".", "?")


def _component(row, category):
    return str(row.get("id" if category == "_chem_comp." else "comp_id", "")).upper()


def _metadata_tables(structure):
    metadata = getattr(structure, "metadata", {}) or {}
    tables = {}
    for category in _SOURCE_CATEGORIES:
        key = category.strip("_.")
        header = metadata.get(key, ())
        values = metadata.get(key + " data", ())
        names = list(header[1:])
        if names and len(values) % len(names) == 0:
            tables[category] = [dict(zip(names, values[start:start + len(names)]))
                                for start in range(0, len(values), len(names))]
    return tables


def _source_tables(structure, warn):
    filename = getattr(structure, "filename", None)
    if not filename:
        return []
    path = Path(filename)
    if not getattr(structure, "is_mmcif", False) and not str(path).lower().endswith(
            (".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        return []
    try:
        import gemmi
        document = gemmi.cif.read_file(str(path))
    except (OSError, ValueError, RuntimeError) as exc:
        warn("Could not retain source chemical dictionaries from %s: %s" % (path, exc))
        return []
    # Keep distinct definitions from different source blocks visible; selecting
    # the first block would silently change the meaning of identical atom names.
    return [{category: _rows(block, category) for category in _SOURCE_CATEGORIES}
            for block in document]


def _compatible(residues, atom_rows, bond_rows):
    """Check named heavy topology, tolerating absent template atoms and added H."""
    named = {}
    for row in atom_rows:
        name, element = row.get("atom_id"), row.get("type_symbol")
        if not _present(name) or not _present(element):
            return "incomplete named atom dictionary"
        element = str(element).upper()
        if name in named and named[name] != element:
            return "conflicting named atom elements"
        named[name] = element
    expected = {frozenset((row.get("atom_id_1"), row.get("atom_id_2"))) for row in bond_rows
                if named.get(row.get("atom_id_1")) not in _HYDROGENS
                and named.get(row.get("atom_id_2")) not in _HYDROGENS}
    for residue in residues:
        atoms = [atom for atom in residue.atoms
                 if str(atom.element.name).upper() not in _HYDROGENS]
        names = {atom.name for atom in atoms}
        if len(names) != len(atoms):
            return "ambiguous live heavy atom names"
        for atom in atoms:
            if named.get(atom.name) != str(atom.element.name).upper():
                return "live atom %s differs from its deposited name/element" % atom.name
        actual = set()
        for atom in atoms:
            if not hasattr(atom, "neighbors"):
                return "live heavy bond topology is unavailable"
            for other in atom.neighbors:
                if other.residue is residue and str(other.element.name).upper() not in _HYDROGENS:
                    actual.add(frozenset((atom.name, other.name)))
        present_expected = {edge for edge in expected if edge.issubset(names)}
        if actual != present_expected:
            return "live heavy bonds differ from the deposited dictionary"
    return None


def _chemical_signature(tables):
    fields = {
        "_chem_comp_atom.": ("atom_id", "type_symbol", "charge", "pdbx_aromatic_flag",
                            "pdbx_stereo_config", "pdbx_leaving_atom_flag"),
        "_chem_comp_bond.": ("atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"),
        "_pdbx_chem_comp_descriptor.": ("type", "descriptor"),
        "_chem_comp.": ("pdbx_smiles",),
    }
    return tuple((category, tuple(sorted(tuple(str(row.get(key)) if _present(row.get(key)) else ""
                                              for key in keys)
                                        for row in tables.get(category, ()))))
                 for category, keys in fields.items())


def _append_rows(block, category, rows):
    current = _rows(block, category)
    seen = {tuple(sorted(row.items())) for row in current}
    for row in rows:
        signature = tuple(sorted(row.items()))
        if signature not in seen:
            current.append(dict(row))
            seen.add(signature)
    names = list(dict.fromkeys(key for row in current for key in row))
    if names:
        block.set_mmcif_category(category, {name: [row.get(name, False) for row in current]
                                           for name in names})


def _supplement_descriptions(block, rows):
    """Fill missing descriptor fields without replacing writer-owned values."""
    current = _rows(block, "_chem_comp.")
    for row in rows:
        existing = [item for item in current if item.get("id") == row.get("id")]
        if not existing:
            current.append(dict(row))
        else:
            for item in existing:
                for key, value in row.items():
                    if not _present(item.get(key)):
                        item[key] = value
    names = list(dict.fromkeys(key for row in current for key in row))
    if names:
        block.set_mmcif_category("_chem_comp.", {name: [row.get(name, False) for row in current]
                                               for name in names})


def preserve_source_chemistry(path, structures, logger=None):
    """Supplement the temporary mmCIF snapshot with compatible named dictionaries.

    Existing output dictionaries are authoritative.  Model metadata takes
    precedence over the original local mmCIF file, component by component.
    Coordinates and explicit connections always remain those of the live export.
    Conflicting definitions across input models remain visible to Chemur's own
    named-chemistry validation instead of being resolved by input ordering.
    """
    import gemmi
    structures = list(structures)
    document = gemmi.cif.read_file(str(path))
    coordinate_blocks = [block for block in document if block.get_mmcif_category("_atom_site.")]
    existing_components = {_component(row, category) for block in document
                           for category in ("_chem_comp_atom.", "_chem_comp_bond.")
                           for row in _rows(block, category)}
    warn = logger.warning if logger is not None else lambda message: None
    # Component dictionaries are global after the coordinate blocks are merged.
    # A template from one model must also fit every same-name residue in the
    # other models, including models with no source dictionary of their own.
    all_residues = defaultdict(list)
    for structure in structures:
        for residue in structure.residues:
            all_residues[str(residue.name).upper()].append(residue)
    changed = False
    definitions = {}
    warned_conflicts = set()
    for index, structure in enumerate(structures):
        residues = defaultdict(list)
        for residue in structure.residues:
            residues[str(residue.name).upper()].append(residue)
        metadata = _metadata_tables(structure)
        metadata_components = {_component(row, "_chem_comp_atom.")
                               for row in metadata.get("_chem_comp_atom.", ())}
        needed = set(residues) - existing_components
        sources = [metadata, *_source_tables(structure, warn)]
        destinations = ([coordinate_blocks[index]] if len(coordinate_blocks) == len(structures)
                        else coordinate_blocks)
        for target in destinations:
            changed = _preserve_residue_provenance(target, structure, sources, index, warn) or changed
        destination = None
        for source_index, tables in enumerate(sources):
            atom_rows = tables.get("_chem_comp_atom.", ())
            components = {_component(row, "_chem_comp_atom.") for row in atom_rows} & needed
            if source_index:
                components -= metadata_components
            for component in sorted(components):
                selected = {category: [row for row in tables.get(category, ())
                                       if _component(row, category) == component]
                            for category in _CATEGORIES}
                reason = _compatible(all_residues[component], selected["_chem_comp_atom."],
                                     selected["_chem_comp_bond."])
                if reason:
                    warn("Chemur did not restore %s chemistry for %s: %s. "
                         "Use a chemistry override for edited ligands." %
                         (component, getattr(structure, "name", "source model"), reason))
                    continue
                signature = _chemical_signature(selected)
                if (component in definitions and definitions[component] != signature
                        and component not in warned_conflicts):
                    warn("Chemur found conflicting deposited dictionaries for %s across source "
                         "models or CIF blocks; the definitions are retained for chemistry "
                         "validation. Use an explicit chemistry override to resolve them." % component)
                    warned_conflicts.add(component)
                definitions.setdefault(component, signature)
                if destination is None:
                    if len(coordinate_blocks) == len(structures):
                        destination = coordinate_blocks[index]
                    else:
                        # The writer can coalesce related models.  A separate
                        # dictionary block is merged by the snapshot normalizer.
                        name = "chemur_source_chemistry_%d" % (index + 1)
                        while document.find_block(name) is not None:
                            name += "_"
                        destination = document.add_new_block(name)
                for category in _CATEGORIES[:-1]:
                    _append_rows(destination, category, selected[category])
                _supplement_descriptions(destination, selected["_chem_comp."])
                changed = True
    if changed:
        document.write_file(str(path))


def _clean(value):
    return str(value) if _present(value) else ""


def _residue_identity(row):
    return (_clean(row.get("auth_asym_id") or row.get("label_asym_id")) or "_",
            _clean(row.get("auth_seq_id") or row.get("label_seq_id")),
            _clean(row.get("pdbx_PDB_ins_code")),
            _clean(row.get("auth_comp_id") or row.get("label_comp_id")))


def _preserve_residue_provenance(block, structure, sources, model_index, warn):
    """Retain deposited role/sequence without restoring any source coordinates.

    Author residue identities locate source provenance; the live scene coordinates
    additionally identify the owning model in a coalesced ChimeraX output block.
    Modified monomers stay inside their original polymer entity and sequence.
    """
    live = {}
    for residue in structure.residues:
        if not hasattr(residue, "number") or not hasattr(residue, "chain_id"):
            continue
        key = (residue.chain_id or "_", str(residue.number),
               str(getattr(residue, "insertion_code", "") or "").strip(), residue.name)
        live.setdefault(key, []).append(residue)
    if not live:
        return False
    candidates = defaultdict(dict)
    for source in sources:
        entries = {_clean(row.get("id")).upper() for row in source.get("_entry.", ())
                   if _present(row.get("id"))}
        if len(entries) != 1:
            continue
        entry = next(iter(entries))
        polymers = {str(row.get("entity_id")): row for row in source.get("_entity_poly.", ())}
        for atom in source.get("_atom_site.", ()):
            key = _residue_identity(atom)
            if key not in live or len(live[key]) != 1:
                continue
            polymer = polymers.get(str(atom.get("label_entity_id")), {})
            sequence = _clean(atom.get("label_seq_id")) if polymer else ""
            if sequence and not sequence.isdigit():
                continue
            signature = (entry, _clean(atom.get("label_asym_id")),
                         _clean(atom.get("label_entity_id")), sequence,
                         _clean(polymer.get("type")))
            candidates[key].setdefault(signature, (source, polymer))
    atom_rows = _rows(block, "_atom_site.")
    matched = {}
    for row in atom_rows:
        key = _residue_identity(row)
        if key not in candidates or len(candidates[key]) != 1:
            continue
        residue = live[key][0]
        name = _clean(row.get("auth_atom_id") or row.get("label_atom_id"))
        element = _clean(row.get("type_symbol")).upper()
        try:
            point = tuple(float(row[axis]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z"))
        except (KeyError, TypeError, ValueError):
            continue
        found = [atom for atom in residue.atoms if atom.name == name
                 and str(atom.element.name).upper() == element
                 and sum((float(x) - y) ** 2 for x, y in zip(
                     getattr(atom, "scene_coord", getattr(atom, "coord", ())), point)) <= 0.05 ** 2
                 and (hasattr(atom, "scene_coord") or hasattr(atom, "coord"))]
        if len(found) == 1:
            matched.setdefault(key, []).append(row)
    if not matched:
        return False
    provenance = []
    group_ids = {}
    label_changes = {}
    for key, rows in matched.items():
        signature, (source, polymer) = next(iter(candidates[key].items()))
        entry, source_asym, source_entity, sequence, polymer_type = signature
        if polymer_type and sequence:
            group_key = (entry, source_asym, source_entity)
            if group_key not in group_ids:
                suffix = "%d_%d" % (model_index + 1, len(group_ids) + 1)
                entity, asym = "chemur_entity_" + suffix, "chemur_chain_" + suffix
                group_ids[group_key] = entity, asym
                _append_rows(block, "_entity.", [{"id": entity, "type": "polymer"}])
                _append_rows(block, "_struct_asym.", [{"id": asym, "entity_id": entity}])
                _append_rows(block, "_entity_poly.", [{**polymer, "entity_id": entity}])
                sequence_rows = [{**record, "entity_id": entity} for record in source.get("_entity_poly_seq.", ())
                                 if str(record.get("entity_id")) == source_entity]
                _append_rows(block, "_entity_poly_seq.", sequence_rows)
            entity, asym = group_ids[group_key]
            for row in rows:
                old = (_clean(row.get("label_asym_id")), _clean(row.get("label_seq_id")),
                       _clean(row.get("label_comp_id")))
                label_changes[old] = (asym, sequence)
                row.update(label_entity_id=entity, label_asym_id=asym, label_seq_id=sequence)
            _append_rows(block, "_pdbx_poly_seq_scheme.", [{
                "asym_id": asym, "entity_id": entity, "seq_id": sequence, "mon_id": key[3],
                "auth_seq_num": key[1], "pdb_seq_num": key[1], "pdb_ins_code": key[2] or False,
                "pdb_strand_id": key[0], "auth_mon_id": key[3], "pdb_mon_id": key[3]}])
        row = rows[0]
        # MODRES parent identity is instance-specific; do not turn it into a
        # global CCD replacement when the same monomer has different products.
        modified = []
        for table in sources:
            for source_record in table.get("_pdbx_struct_mod_residue.", ()):
                source_key = (_clean(source_record.get("auth_asym_id")) or "_",
                              _clean(source_record.get("auth_seq_id")),
                              _clean(source_record.get("PDB_ins_code")),
                              _clean(source_record.get("auth_comp_id")))
                if source_key != key:
                    continue
                restored = dict(source_record)
                restored.update(label_asym_id=row.get("label_asym_id", False),
                                label_seq_id=row.get("label_seq_id", False),
                                label_comp_id=row.get("label_comp_id", key[3]))
                restored["id"] = "chemur_mod_%d_%s_%s_%s%s" % (
                    model_index + 1, key[0], key[3], key[1], key[2])
                modified.append(restored)
        _append_rows(block, "_pdbx_struct_mod_residue.", modified)
        provenance.append({"label_asym_id": row.get("label_asym_id", False),
            "label_seq_id": row.get("label_seq_id", False), "auth_asym_id": key[0],
            "auth_seq_id": key[1] + key[2], "pdbx_PDB_ins_code": key[2] or False,
            "comp_id": key[3], "source_entry_id": entry,
            "source_polymer_type": polymer_type or False})
    # Retain live bonds after any restored label/sequence identifiers.
    connections = _rows(block, "_struct_conn.")
    for connection in connections:
        for prefix in ("ptnr1_", "ptnr2_"):
            key = tuple(_clean(connection.get(prefix + field))
                        for field in ("label_asym_id", "label_seq_id", "label_comp_id"))
            if key in label_changes:
                connection[prefix + "label_asym_id"], connection[prefix + "label_seq_id"] = label_changes[key]
    for category, rows in (("_atom_site.", atom_rows), ("_struct_conn.", connections)):
        names = list(dict.fromkeys(name for row in rows for name in row))
        if names:
            block.set_mmcif_category(category, {name: [row.get(name, False) for row in rows] for name in names})
    _append_rows(block, "_chemur_source_residue.", provenance)
    entries = {row["source_entry_id"] for row in _rows(block, "_chemur_source_residue.")}
    if len(entries) == 1:
        block.set_pair("_entry.id", next(iter(entries)))
    return True
