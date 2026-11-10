"""Package and publish the shared Retinexformer demo to Hugging Face Spaces."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, SpaceHardware
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]


def prepare_bundle(destination, hardware):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        "app.py", "requirements-demo.txt", "basicsr/models/archs/RetinexFormer_arch.py",
        "Options/RetinexFormer_LOL_v1.yml", "figure/inference_demo.png",
    ]
    files.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "demo/examples").iterdir()) if path.is_file())
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in ["space_app.py", "space_success_stats.py", "README.md", "requirements.txt"]:
        shutil.copy2(ROOT / "demo/huggingface" / name, destination / name)
    # Spaces mounts requirements.txt alone under /tmp during its dependency
    # build, so nested relative -r files must be expanded before uploading.
    path = destination / "requirements.txt"
    requirements = path.read_text().replace(
        "-r requirements-demo.txt\n", (ROOT / "requirements-demo.txt").read_text()
    )
    if hardware == "cpu-basic":
        requirements = "--extra-index-url https://download.pytorch.org/whl/cpu\n" + requirements.replace(
            "torch==2.9.1\n", "torch==2.9.1+cpu\n"
        )
    path.write_text(requirements)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Defaults to <logged-in-user>/Retinexformer")
    parser.add_argument("--hardware", choices=["zero-a10g", "cpu-basic"])
    parser.add_argument("--update", action="store_true", help="Update an existing Space")
    parser.add_argument("--prepare-only", type=Path, help="Build a deployable directory without publishing")
    args = parser.parse_args()
    if args.prepare_only:
        print(prepare_bundle(args.prepare_only, args.hardware or "zero-a10g"))
        return
    api = HfApi()
    user = api.whoami()
    repo_id = args.repo_id or f"{user['name']}/Retinexformer"
    exists = True
    try:
        info = api.space_info(repo_id)
    except RepositoryNotFoundError:
        exists = False
    if exists and not args.update:
        parser.error(f"Space {repo_id} exists; use --update to deploy changes.")
    hardware = args.hardware or "zero-a10g"
    if exists:
        current = info.runtime.requested_hardware or info.runtime.hardware
        if args.hardware and args.hardware != current:
            api.request_space_hardware(repo_id, SpaceHardware(args.hardware))
        else:
            hardware = current or hardware
    if not exists:
        try:
            api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=False,
                            space_hardware=SpaceHardware(hardware))
        except HfHubHTTPError as error:
            if hardware != "zero-a10g" or error.response.status_code not in (400, 403):
                raise
            print("ZeroGPU is unavailable for this account; using free CPU Basic.")
            hardware = "cpu-basic"
            api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=False,
                            space_hardware=SpaceHardware.CPU_BASIC, exist_ok=True)
    with TemporaryDirectory(prefix="retinexformer-space-") as folder:
        prepare_bundle(folder, hardware)
        commit = api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=folder,
                                   commit_message="Deploy Retinexformer interactive inference demo")
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"Hardware: {hardware}")
    print(f"Commit: {commit.oid}")


if __name__ == "__main__":
    main()
