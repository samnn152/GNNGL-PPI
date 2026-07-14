"""Download and install the external MASSA embedding artifact."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


GNN_PPI_EMBEDDING_ID = "1sq2VQGAMWmWg02hqhyWju2xuiJ-oHbq0"
MASSA_SOURCE_NAME = "shs_our_model_all.pickle"
MASSA_TARGET_NAME = "shs_MASSA.pickle"


class MassaPretrainer:
    """Download, validate, extract, and install pretrained MASSA embeddings."""

    @staticmethod
    def download_archive(output_path: Path) -> Path:
        """Download the published embedding archive unless it already exists."""
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

    @staticmethod
    def extract_archive(archive_path: Path, output_dir: Path) -> Path:
        """Safely extract the archive and return its expected embedding file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_file = output_dir / "pretrained_emb" / MASSA_SOURCE_NAME
        if expected_file.exists():
            return expected_file

        with tarfile.open(archive_path, "r:gz") as archive:
            root = output_dir.resolve()
            for member in archive.getmembers():
                destination = (output_dir / member.name).resolve()
                if root not in destination.parents and destination != root:
                    raise ValueError(f"Unsafe archive member: {member.name}")
            archive.extractall(output_dir)
        return expected_file

    @staticmethod
    def install_massa_embedding(source_path: Path, target_dir: Path) -> Path:
        """Copy the extracted embedding to the runtime asset location."""
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / MASSA_TARGET_NAME
        shutil.copy2(source_path, target_path)
        return target_path

    @classmethod
    def prepare(cls, download_dir: str | Path = "downloads", target_dir: str | Path = "assets/pretrained") -> Path:
        """Run the idempotent download-to-install preparation workflow."""
        download_dir = Path(download_dir)
        target_dir = Path(target_dir)

        archive_path = cls.download_archive(download_dir / "gnn_ppi_pretrained_embedding.tar.gz")
        source_path = cls.extract_archive(archive_path, download_dir)
        return cls.install_massa_embedding(source_path, target_dir)


class MassaPretrainCLI:
    """Command-line adapter for preparing MASSA embedding assets."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Create the standalone pretraining-asset argument parser."""
        parser = argparse.ArgumentParser(description="Download and install MASSA GNN-PPI embeddings.")
        parser.add_argument("--download_dir", default="downloads")
        parser.add_argument("--target_dir", default="assets/pretrained")
        return parser

    @classmethod
    def run(cls) -> None:
        """Parse CLI arguments, prepare the embedding, and print its path."""
        args = cls.build_parser().parse_args()
        target_path = MassaPretrainer.prepare(args.download_dir, args.target_dir)
        print("Installed MASSA embedding: {}".format(target_path))


if __name__ == "__main__":
    MassaPretrainCLI.run()
