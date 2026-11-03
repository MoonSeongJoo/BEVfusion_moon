import argparse
import json
import numpy as np


def calculate_error(estimated, ground_truth):

    relative = (
        estimated @ np.linalg.inv(ground_truth)
    )

    R = relative[:3, :3]
    t = relative[:3, 3]

    cos_angle = np.clip(
        (np.trace(R) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    rot_error = np.degrees(
        np.arccos(cos_angle)
    )

    trans_error = np.linalg.norm(t)

    return rot_error, trans_error


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("manifest")

    parser.add_argument(
        "--expect-frames",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    records = manifest["records"]

    print("Total frames:", len(records))

    if (
        args.expect_frames
        and len(records) != args.expect_frames
    ):
        raise RuntimeError(
            "Unexpected frame count"
        )

    broken_errors = []
    corrected_errors = []

    for record in records.values():

        clean = np.asarray(
            record["clean_cam2lidar"]
        )

        broken = np.asarray(
            record["broken_cam2lidar"]
        )

        corrected = np.asarray(
            record["corrected_cam2lidar"]
        )

        for cam in range(6):

            broken_errors.append(
                calculate_error(
                    broken[cam],
                    clean[cam],
                )
            )

            corrected_errors.append(
                calculate_error(
                    corrected[cam],
                    clean[cam],
                )
            )

    broken_errors = np.asarray(broken_errors)
    corrected_errors = np.asarray(corrected_errors)

    for name, values in [
        ("BROKEN", broken_errors),
        ("LGPC", corrected_errors),
    ]:

        print(f"\n[{name}]")

        print(
            "Rotation Mean:",
            values[:, 0].mean(),
        )

        print(
            "Translation Mean:",
            values[:, 1].mean(),
        )

        print(
            "Rotation P90:",
            np.percentile(values[:, 0], 90),
        )

        print(
            "Translation P90:",
            np.percentile(values[:, 1], 90),
        )

    rot_recovery = 100 * (
        1.0
        - corrected_errors[:, 0].mean()
        / broken_errors[:, 0].mean()
    )

    trans_recovery = 100 * (
        1.0
        - corrected_errors[:, 1].mean()
        / broken_errors[:, 1].mean()
    )

    print("\nRotation Recovery:", rot_recovery, "%")
    print("Translation Recovery:", trans_recovery, "%")


if __name__ == "__main__":
    main()