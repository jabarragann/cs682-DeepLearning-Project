from pathlib import Path
import dotenv
import os

# Changes to config class
class Config:
    # Load data paths as environment
    module_root = Path(__file__)
    config_path = Path(module_root.parent) / ".project_paths"
    if not config_path.exists():
        print("Create the configuration file at deepgesture/.project_paths")
        exit(0)

    dotenv.load_dotenv(config_path)
    suturing_raw_dir = Path(os.getenv("SUTURING_RAW_DIR"))
    suturing_processed_dir = Path(os.getenv("SUTURING_PROCESSED_DIR"))
    # ------------------------------------------------------------
    # Suturing paths
    # ------------------------------------------------------------
    # Raw paths
    suturing_transcriptions_dir = suturing_raw_dir / "transcriptions"
    suturing_videos_dir = suturing_raw_dir / "video"
    suturing_kinematics_dir = suturing_raw_dir / "kinematics/AllGestures"
    # Processing paths
    optical_flow_dir = suturing_processed_dir / "OpticalFlow"
    blobs_dir = suturing_processed_dir / "blobs"

    # ------------------------------------------------------------
    # Knot-tying paths
    # ------------------------------------------------------------

    # Models
    trained_models_dir = suturing_processed_dir / "models"


def check_paths():
    paths = [
        Config.suturing_raw_dir,
        Config.suturing_processed_dir,
        Config.optical_flow_dir,
        Config.suturing_videos_dir,
        Config.suturing_kinematics_dir,
    ]

    checks = []
    for p in paths:
        checks.append(p.exists())
        print(f"{p}: {p.exists()}")

    print(f"All path are correct: {all(checks)}")


if __name__ == "__main__":
    check_paths()
