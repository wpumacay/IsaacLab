# IsaacSim|IsaacLab initialization -------------------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Utility to convert a MJCF into USD format.")
parser.add_argument(
    "--action",
    type=str,
    choices=[
        "clean-lightwheel",
        "resize-thor-assets-single",
        "resize-thor-assets",
        "add-freejoint-single",
        "add-freejoint",
        "remove-sites-single",
        "remove-sites",
        "convert-to-usd-single",
        "convert-to-usd",
        "collect-usd-single",
        "collect-usd",
        "convert-to-mjcf",
    ],
)
parser.add_argument("--folder", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--inplace", action="store_true")
parser.add_argument("--overwrite", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------------------------------------------------

import asyncio
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

import mujoco as mj
import numpy as np
import omni.kit.commands
import omni.usd
import trimesh
import tyro
import yaml
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaaclab.utils.dict import print_dict
from isaacsim.core.utils import extensions
from tqdm import tqdm

extensions.enable_extension("isaacsim.asset.importer.mjcf")
extensions.enable_extension("omni.kit.usd.collect")

from omni.kit.usd.collect import Collector

ROOT_DIR = Path(__file__).parent.parent.parent
USD_PROCESSED_DIR = ROOT_DIR / "assets" / "objects" / "thor" / "usd-processed"

@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            # restore stdout. buffering and flags such as CLOEXEC may be different
            _redirect_stdout(to=old_stdout)


THOR_ASSETS_IDS_TO_SKIP: set[str] = {"Light_Switch", "RoboTHOR_dresser_aneboda"}


@dataclass
class Args:
    action: Literal[
        "clean-lightwheel",
        "resize-thor-assets-single",
        "resize-thor-assets",
        "add-freejoint-single",
        "add-freejoint",
        "convert-to-usd-single",
        "convert-to-usd",
        "collect-usd-single",
        "collect-usd",
        "convert-to-mjcf",
    ]
    folder: Path | None = None  # the path to the folder containing the assets to be processed
    model: Path | None = None
    inplace: bool = False
    headless: bool = False
    overwrite: bool = False


def clean_lightwheel_single_mjcf(model_path: Path) -> None:
    spec = mj.MjSpec.from_file(model_path.as_posix())

    def_vis: mj.MjsDefault | None = spec.find_default("visual")
    if def_vis is not None:
        def_vis.geom.group = 0

    def_col: mj.MjsDefault | None = spec.find_default("collision")
    if def_col is not None:
        def_col.geom.group = 4

    region_geoms: list[mj.MjsGeom] = []
    for geom in spec.geoms:
        assert isinstance(geom, mj.MjsGeom)
        if geom.classname.name == "visual":
            geom.group = 0
        elif geom.classname.name == "collision":
            geom.group = 4
        elif geom.classname.name == "region":
            region_geoms.append(geom)

    for geom in region_geoms:
        geom.delete()

    _ = spec.compile()
    with open(model_path, "w") as fhandle:
        fhandle.write(spec.to_xml())


def clean_lightwheel(args: Args, blocks_metadata: Any) -> None:
    for data in tqdm(blocks_metadata):
        name = data.get("name", "")
        ptype = data.get("type", "")
        if name == "" or ptype == "":
            continue
        if ptype != "mjcf":
            continue
        model_path = args.folder / name / "model.xml"
        if not model_path.is_file():
            continue
        try:
            clean_lightwheel_single_mjcf(model_path)
        except Exception as e:
            print(f"[ERROR]: got an error while working with '{model_path.stem}', error: {e}")


def resize_thor_assets_single(model_path: Path, check_if_need_to: bool = False, inplace: bool = False) -> bool:
    modified = False
    try:
        # if True:
        spec = mj.MjSpec.from_file(model_path.as_posix())

        for geom in spec.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.type != mj.mjtGeom.mjGEOM_MESH:
                continue
            if geom.classname.name != "__DYNAMIC_MJT__":
                continue
            mesh_handle = spec.mesh(geom.meshname)
            if mesh_handle is None:
                continue
            mesh_path = model_path.parent / mesh_handle.file
            if not mesh_path.is_file():
                continue

            # Don't scale if the scale is close to (1, 1, 1)
            if check_if_need_to and np.allclose(mesh_handle.scale, np.ones_like(mesh_handle.scale), atol=1e-3):
                continue
            modified = True

            mesh = trimesh.load_mesh(mesh_path.as_posix())
            if isinstance(mesh, list):
                mesh = mesh[0]
            mesh.apply_scale(mesh_handle.scale)

            new_mesh_path = mesh_path.parent / f"{mesh_path.stem}_fix.obj" if not inplace else mesh_path

            with open(new_mesh_path, "w") as fhandle:
                trimesh.exchange.export.export_mesh(mesh, fhandle, file_type="obj")

            mesh_handle.file = new_mesh_path.relative_to(model_path.parent).as_posix()
            mesh_handle.scale = np.ones_like(mesh_handle.scale)

        if modified:
            new_model_path = model_path.parent / f"{model_path.stem}_sc.xml" if not inplace else model_path
            _ = spec.compile()
            with open(new_model_path, "w") as fhandle:
                fhandle.write(spec.to_xml())

    except Exception as e:
        print(f"[ERROR]: couldn't resize thor assets '{model_path.stem}', error: {e}")

    return modified


def resize_thor_assets(folder_path: Path, inplace: bool = False) -> None:
    models_filepaths: list[Path] = []
    for candidate_xml in folder_path.rglob("*.xml"):
        if any(substr in candidate_xml.stem for substr in ("_old", "_fix", "_upt", "_orig", "_sc")):
            continue
        if "_mesh" not in candidate_xml.stem:
            continue
        models_filepaths.append(candidate_xml)

    models_modified: list[str] = []
    for model_path in tqdm(models_filepaths):
        modified = resize_thor_assets_single(model_path, check_if_need_to=True, inplace=inplace)
        if modified:
            models_modified.append(model_path.stem)

    print("Model modified")
    print("-" * 80)
    pprint(models_modified)
    print("-" * 80)

def add_freejoint_single(model_path: Path) -> None:
    try:
        spec = mj.MjSpec.from_file(model_path.as_posix())
        if all(jnt.type != mj.mjtJoint.mjJNT_FREE for jnt in spec.joints):
            root_handle = spec.worldbody.first_body()
            if root_handle is not None:
                root_handle.add_freejoint(name=f"{spec.modelname}_free")
                _ = spec.compile()
                with open(model_path, "w") as fhandle:
                    fhandle.write(spec.to_xml())
    except Exception as e:
        print(f"[ERROR]: couldn't add freejoint to model '{model_path.stem}', error: {e}")

def add_freejoint(folder_path: Path) -> None:
    models_filepaths: list[Path] = []
    for candidate_xml in folder_path.rglob("*.xml"):
        if any(substr in candidate_xml.stem for substr in ("_old", "_fix", "_upt", "_orig", "_sc")):
            continue
        if any(substr in candidate_xml.stem for substr in THOR_ASSETS_IDS_TO_SKIP):
            continue
        models_filepaths.append(candidate_xml)

    models_modified: list[str] = []
    for model_path in tqdm(models_filepaths):
        modified = add_freejoint_single(model_path)
        if modified:
            models_modified.append(model_path.stem)

    print("Model modified")
    print("-" * 80)
    pprint(models_modified)
    print("-" * 80)

def convert_to_usd_single(model_path: Path) -> None:
    if not model_path.is_absolute():
        model_path = model_path.absolute()

    model_usd_path = model_path.parent / f"{model_path.stem}.usda"

    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=model_path.as_posix(),
        usd_dir=model_usd_path.parent.as_posix(),
        usd_file_name=model_usd_path.name,
        fix_base=False,
        import_sites=False,
        force_usd_conversion=True,
        make_instanceable=False,
    )

    print("-" * 80)
    print("-" * 80)
    print(f"Input MJCF file: {model_path}")
    print("MJCF importer config:")
    print_dict(mjcf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    mjcf_converter = MjcfConverter(mjcf_converter_cfg)

    print("MJCF importer output:")
    print(f"Generated USD file: {mjcf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)


def convert_to_usd(folder_path: Path, overwrite: bool = False) -> None:
    models_filepaths: list[Path] = []
    for candidate_xml in folder_path.rglob("*.xml"):
        if any(substr in candidate_xml.stem for substr in ("_old", "_fix", "_upt", "_orig", "_sc")):
            continue
        if any(substr in candidate_xml.stem for substr in THOR_ASSETS_IDS_TO_SKIP):
            continue
        model_usd_path = candidate_xml.parent / f"{candidate_xml.stem}.usda"
        if not overwrite and model_usd_path.is_file():
            continue
        models_filepaths.append(candidate_xml)

    for model_path in tqdm(models_filepaths):
        convert_to_usd_single(model_path)

def collect_usd_single(model_path: Path) -> None:
    global simulation_app
    assert simulation_app is not None, "SimulationApp global instance should be initialized by now"

    success = simulation_app.context.open_stage(model_path.as_posix())
    if not success:
        return

    async def run_collection_async():
        collector = Collector(
            usd_path=model_path.as_posix(),
            collect_dir=USD_PROCESSED_DIR.as_posix(),
        )
        return await collector.collect()

    loop = asyncio.get_event_loop()

    collection_task = loop.create_task(run_collection_async())

    while not collection_task.done():
        simulation_app.update()

    try:
        result = collection_task.result()
        if isinstance(result, tuple):
            success = result[0]
            output_path = result[1] if len(result) > 1 else "Unknown"
        else:
            success = result
            output_path = model_path.parent.as_posix()

        if success:
            print(f"[SUCCESS] Collection complete. Saved to: {output_path}")
        else:
            print(f"[ERROR] Collection failed. Return value: {result}")
    except Exception as e:
        print(f"[ERROR] Exception during collection: {e}")

    simulation_app.context.close_stage()



def main() -> int:
    args = tyro.cli(Args)

    match args.action:
        case "clean-lightwheel":
            if args.folder is None or not args.folder.is_dir():
                print(f"[ERROR]: the folder path '{args.folder}' is not a valid directory")
                return 1
            metadata_filepath = args.folder / "metadata.yaml"
            with open(metadata_filepath, "r") as fhandle:
                metadata = yaml.safe_load(fhandle)
            if "blocks" in metadata:
                clean_lightwheel(args, metadata["blocks"])
        case "resize-thor-assets-single":
            if args.model is None or not args.model.is_file():
                print(f"[ERROR]: the model path '{args.model}' is not a valid file")
                return 1
            resize_thor_assets_single(args.model, inplace=args.inplace)
        case "resize-thor-assets":
            if args.folder is None or not args.folder.is_dir():
                print(f"[ERROR]: the model path '{args.folder}' is not a valid directory")
                return 1
            resize_thor_assets(args.folder, inplace=args.inplace)
        case "add-freejoint-single":
            if args.model is None or not args.model.is_file():
                print(f"[ERROR]: the model path '{args.model}' is not a valid file")
                return 1
            add_freejoint_single(args.model)
        case "add-freejoint":
            if args.folder is None or not args.folder.is_dir():
                print(f"[ERROR]: the folder path '{args.folder}' is not a valid directory")
                return 1
            add_freejoint(args.folder)
        case "convert-to-usd-single":
            if args.model is None or not args.model.is_file():
                print(f"[ERROR]: the model path '{args.model}' is not a valid file")
                return 1
            convert_to_usd_single(args.model)
        case "convert-to-usd":
            if args.folder is None or not args.folder.is_dir():
                print(f"[ERROR]: the folder path '{args.folder}' is not a valid directory")
                return 1
            convert_to_usd(args.folder, args.overwrite)
        case "collect-usd-single":
            if args.model is None or not args.model.is_file():
                print(f"[ERROR]: the model path '{args.model}' is not a valid file")
                return 1
            collect_usd_single(args.model)
        case "convert-to-mjcf":
            if args.folder is None or not args.folder.is_dir():
                print(f"[ERROR]: the folder path '{args.folder}' is not a valid directory")
                return 1
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
