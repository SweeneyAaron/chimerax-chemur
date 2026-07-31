"""ChimeraX-Chemur: predict and visualize biomolecular interactions with Chemur."""

from chimerax.core.toolshed import BundleAPI


class _ChemurBundleAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        # Called once per registered command (here, "chemur interactions").
        from . import cmd
        cmd.register_command(ci.name, logger)

    @staticmethod
    def start_tool(session, bi, ti):
        # Called when the user launches the GUI tool from the Tools menu.
        from .tool import ChemurTool
        return ChemurTool(session, ti.name)


bundle_api = _ChemurBundleAPI()
