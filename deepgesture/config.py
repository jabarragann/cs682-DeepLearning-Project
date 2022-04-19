from pathlib import Path


class Config:

    suturing_raw_dir = Path("/home/juan1995/research_juan/CS682/jigsaw-dataset/Suturing-Raw")
    suturing_processed_dir = Path("/home/juan1995/research_juan/CS682/jigsaw-dataset/Suturing-Processed")

    optical_flow_dir = suturing_processed_dir / "OpticalFlow"
    suturing_videos_dir = suturing_raw_dir / "video"
    suturing_kinematics_dir = suturing_raw_dir / "kinematics/AllGestures"


if __name__ == "__main__":

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
