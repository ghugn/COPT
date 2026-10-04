import os
import sys
import shutil
import argparse
import time
from pathlib import Path

# Ensure UTF-8 stdout/stderr
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parent.parent)

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets (e.g. RB-small) and prepare for training/download")
    parser.add_argument("--format", type=str, default="rb", help="Dataset format (default: rb)")
    parser.add_argument("--name", type=str, default="small", help="Dataset name/size (default: small)")
    parser.add_argument("--num_samples", type=int, default=6000, help="Number of graphs to generate (paper default: 6000)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of multiprocessing workers (default: 2)")
    parser.add_argument("--no_multiprocessing", action="store_true", help="Disable multiprocessing")
    parser.add_argument("--download", action="store_true", help="Trigger google.colab.files.download if in Colab")
    parser.add_argument("--zip", action="store_true", help="Create a .zip archive of the processed data for easy transfer")
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / args.format / args.name / "processed"
    data_file = processed_dir / "data.pt"

    print("=" * 70)
    print(f"GENERATING {args.num_samples} GRAPHS FOR: {args.format.upper()}-{args.name}")
    print("=" * 70)
    print(f"Target location: {data_file}")

    # Remove existing processed file/directory to force generation
    if processed_dir.exists():
        print(f"Removing old processed cache at {processed_dir} to generate fresh data...")
        shutil.rmtree(processed_dir)

    use_mp = not args.no_multiprocessing
    print(f"Multiprocessing: {use_mp} (workers: {args.num_workers})")
    print(f"Generating {args.num_samples} graphs... (This may take a while, progress will be shown)")

    t0 = time.time()

    # Import dependencies
    from src.data.synthetic_datamodule import SyntheticDataModule

    if args.format == "rb":
        dm = SyntheticDataModule(
            data_dir="data",
            format="rb",
            name=args.name,
            task="maxclique",
            batch_size=32,
            splits="5-fold",
            labels=False,
            graph_stats=['degree', 'cluster_coefficient', 'triangle_count'],
            num_workers=args.num_workers if use_mp else 0,
            pin_memory=False,
            multiprocessing=use_mp,
            num_samples=args.num_samples,
            n=[200, 300],
            na=[20, 25],
            k=[5, 12],
        )
    elif args.format == "ba":
        dm = SyntheticDataModule(
            data_dir="data",
            format="ba",
            name=args.name,
            task="maxclique",
            batch_size=32,
            splits="5-fold",
            labels=False,
            graph_stats=['degree', 'cluster_coefficient', 'triangle_count'],
            num_workers=args.num_workers if use_mp else 0,
            pin_memory=False,
            multiprocessing=use_mp,
            num_samples=args.num_samples,
            n=[200, 300],
            m=4,
        )
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    # Run data preparation (graph generation + pre_transforms + saving data.pt)
    dm.prepare_data()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"SUCCESS: Generated {args.num_samples} graphs in {elapsed:.1f}s ({elapsed/60:.1f} mins)!")
    print(f"Saved to: {data_file}")

    if data_file.exists():
        import torch
        d = torch.load(data_file, weights_only=False)
        num_g = len(d[1]['edge_index']) - 1 if isinstance(d, tuple) else len(d)
        print(f"Verified dataset content: {num_g} graphs successfully saved.")

    # Create zip archive if requested
    zip_path = PROJECT_ROOT / f"data_{args.format}_{args.name}.zip"
    if args.zip or args.download:
        print(f"\nCompressing {processed_dir} into {zip_path}...")
        shutil.make_archive(str(PROJECT_ROOT / f"data_{args.format}_{args.name}"), 'zip', processed_dir)
        print(f"Created: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

    # Download in Colab if requested
    if args.download:
        try:
            from google.colab import files
            target_to_dl = str(zip_path if zip_path.exists() else data_file)
            print(f"\nTriggering Colab browser download for: {target_to_dl}...")
            files.download(target_to_dl)
            print("Download triggered in your browser!")
        except Exception as e:
            print(f"Colab download could not be triggered automatically: {e}")
            print(f"You can download manually with:")
            print(f"from google.colab import files; files.download('{data_file}')")

    print("=" * 70)

if __name__ == "__main__":
    main()
