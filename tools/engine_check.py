"""Report which chemur engine ChimeraX actually loaded, and whether its C++ core is active.

Run from the repo root, inside ChimeraX:

    ChimeraX --nogui --exit --cmd "runscript tools/engine_check.py"

ChimeraX has no `shell` command (Shell is a Tool, not a command), and running
`<ChimeraX>/bin/python3.11` directly can resolve a different site-packages than
ChimeraX loads -- `runscript` is the only check that reflects the real environment.
"""

import chemur
from chemur.core import USING_CPP_CORE

print("chemur", getattr(chemur, "__version__", "?"), "at", chemur.__file__)
print("USING_CPP_CORE =", USING_CPP_CORE)
