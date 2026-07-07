import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


GNN_PPI_EMBEDDING_ID = "1sq2VQGAMWmWg02hqhyWju2xuiJ-oHbq0"
MASSA_SOURCE_NAME = "shs_our_model_all.pickle"
MASSA_TARGET_NAME = "shs_MASSA.pickle"


def download_archive(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    subprocess.run(
        [
            "python",
            "-m",
            "gdown",
            GNN_PPI_EMBEDDING_ID,
            "-O",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def extract_archive(archive_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_file = output_dir / "pretrained_emb" / MASSA_SOURCE_NAME
    if expected_file.exists():
        return expected_file

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(output_dir)
    return expected_file


def install_massa_embedding(source_path, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / MASSA_TARGET_NAME
    shutil.copy2(source_path, target_path)
    return target_path


def prepare_massa(download_dir="downloads", target_dir="pre_train_data"):
    download_dir = Path(download_dir)
    target_dir = Path(target_dir)

    archive_path = download_archive(download_dir / "gnn_ppi_pretrained_embedding.tar.gz")
    source_path = extract_archive(archive_path, download_dir)
    return install_massa_embedding(source_path, target_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and install MASSA GNN-PPI embeddings.")
    parser.add_argument("--download_dir", default="downloads")
    parser.add_argument("--target_dir", default="pre_train_data")
    args = parser.parse_args()

    target_path = prepare_massa(args.download_dir, args.target_dir)
    print("Installed MASSA embedding: {}".format(target_path))


if __name__ == "__main__":
    main()
