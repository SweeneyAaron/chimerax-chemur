"""ChimeraX-ChemeleonX: predict and visualize biomolecular interactions with ChemeleonX."""

from chimerax.core.toolshed import BundleAPI


class _ChemeleonXBundleAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        # Called once per registered command (here, "chemeleonx interactions").
        from . import cmd
        cmd.register_command(ci.name, logger)

    @staticmethod
    def start_tool(session, bi, ti):
        # Called when the user launches the GUI tool from the Tools menu.
        from .tool import ChemeleonXTool
        return ChemeleonXTool(session, ti.name)


bundle_api = _ChemeleonXBundleAPI()
