from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco as mj
import numpy as np
import trimesh
import tyro
import yaml
from tqdm import tqdm

@dataclass
class Args:
    action: Literal["clean-lightwheel", "resize-thor-assets-single", "convert-to-usd", "convert-to-mjcf"]
    flavor: Literal["blocks"] = "blocks" # the assets type to be processed (currently only lightwheel-blocks)
    folder: Path | None = None # the path to the folder containing the assets to be processed
    model: Path | None = None

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

def resize_thor_assets_single(model_path: Path) -> None:
    # try:
    if True:
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
            mesh = trimesh.load_mesh(mesh_path.as_posix())
            mesh.apply_scale(mesh_handle.scale)

            new_mesh_path = mesh_path.parent / f"{mesh_path.stem}_fix.obj"
            with open(new_mesh_path, "w") as fhandle:
                trimesh.exchange.export.export_mesh(mesh, fhandle, file_type="obj")

            mesh_handle.file = new_mesh_path.relative_to(model_path.parent).as_posix()
            mesh_handle.scale = np.ones_like(mesh_handle.scale)

        new_model_path = model_path.parent / f"{model_path.stem}_sc.xml"
        _ = spec.compile()
        with open(new_model_path, "w") as fhandle:
            fhandle.write(spec.to_xml())

    # except Exception as e:
    #     print(f"[ERROR]: couldn't resize thor assets '{model_path.stem}', error: {e}")

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
            resize_thor_assets_single(args.model)
        case "convert-to-usd":
            pass
        case "convert-to-mjcf":
            pass

    return 0




if __name__ == "__main__":
    raise SystemExit(main())





