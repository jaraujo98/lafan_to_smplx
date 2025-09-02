"""
MIT License

Copyright (c) 2025 Joao Pedro Araujo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan1 import (
    load_lafan1_file as _load_lafan1_file,
)


def load_lafan1_file(file_path):
    return _load_lafan1_file(file_path)[0]


def get_smplx_local_joint(
    lafan_frame_name, lafan_frame_quat, lafan_parent_frame_name, lafan_parent_frame_quat
):
    parent_frame_global_rot = (
        R.from_quat(lafan_parent_frame_quat, scalar_first=True)
        * lafan_frame_offsets[lafan_parent_frame_name]
    )
    frame_global_quat = (
        R.from_quat(lafan_frame_quat, scalar_first=True)
        * lafan_frame_offsets[lafan_frame_name]
    )
    frame_local_quat = parent_frame_global_rot.inv() * frame_global_quat
    return frame_local_quat.as_rotvec()


lafan_frame_offsets = {
    "world": R.from_euler("x", 0),
    "Hips": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "LeftUpLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "LeftLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "LeftFoot": R.from_euler("z", 0.37117860986509) * R.from_euler("y", np.pi / 2),
    "LeftToe": R.from_euler("y", np.pi / 2),
    "RightUpLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "RightLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "RightFoot": R.from_euler("z", 0.37117860986509) * R.from_euler("y", np.pi / 2),
    "RightToe": R.from_euler("y", np.pi / 2),
    "Spine": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Spine1": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Spine2": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Neck": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Head": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "LeftShoulder": R.from_euler("x", np.pi / 2),
    "LeftArm": R.from_euler("x", np.pi / 2),
    "LeftForeArm": R.from_euler("x", np.pi / 2),
    "LeftHand": R.from_euler("x", np.pi / 2),
    "RightShoulder": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightArm": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightForeArm": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightHand": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
}

smplx_to_lafan_map = {
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "spine1": "Spine",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "spine2": "Spine1",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Spine2",
    "left_foot": "LeftToe",
    "right_foot": "RightToe",
    "neck": "Neck",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}


betas = {
    "smpl": torch.tensor(
        [
            [
                0.9597,
                1.0887,
                -2.1717,
                -0.8611,
                1.3940,
                0.1401,
                -0.2469,
                0.3182,
                -0.2482,
                0.3085,
            ]
        ],
        dtype=torch.float32,
    ),
    "smplx": torch.tensor(
        [
            [
                1.4775,
                0.6674,
                -1.1742,
                0.4731,
                1.2984,
                -0.2159,
                1.5276,
                -0.3152,
                -0.6441,
                -0.2986,
                0.5089,
                -0.6354,
                0.3321,
                -0.1099,
                -0.3060,
                -0.7330,
            ]
        ],
        dtype=torch.float32,
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        type=str,
    )
    parser.add_argument(
        "--smplx_model_path",
        type=str,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="smplx",
        choices=["smpl", "smplx"],
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the output .npz file",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Whether to visualize the result in Rerun",
    )
    args = parser.parse_args()

    # Load the LAFAN BVH file
    data = read_bvh(args.bvh_file)
    frames = load_lafan1_file(args.bvh_file)
    n_frames = len(frames)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model = smplx.create(
        model_path=args.smplx_model_path,
        model_type=args.model_type,
        use_pca=False,
        betas=betas[args.model_type].to(device),
    )

    if args.rerun:
        import rerun as rr

        rr.init("lafan_to_smplx_naive", spawn=True)

    trans_tensor = torch.zeros((n_frames, 3), device=device, dtype=torch.float32)
    root_orient_tensor = torch.zeros((n_frames, 3), device=device, dtype=torch.float32)
    body_pose_tensor = torch.zeros(
        (n_frames, body_model.NUM_BODY_JOINTS, 3), device=device, dtype=torch.float32
    )
    for frame in range(n_frames):
        # Get the root orientation and body pose for the current frame
        root_orient = (
            R.from_quat(frames[frame]["Hips"][1], scalar_first=True)
            * lafan_frame_offsets["Hips"]
        )

        body_pose = torch.zeros(
            (body_model.NUM_BODY_JOINTS, 3), device=device, dtype=torch.float32
        )
        for i, joint_name in enumerate(
            JOINT_NAMES[1:22]
        ):  # Joint names for SMPL and SMPL-X are equal
            if joint_name not in smplx_to_lafan_map:
                continue
            lafan_joint_name = smplx_to_lafan_map[joint_name]
            parent_name = data.bones[data.parents[data.bones.index(lafan_joint_name)]]
            body_pose[i] = torch.from_numpy(
                get_smplx_local_joint(
                    lafan_joint_name,
                    frames[frame][lafan_joint_name][1],
                    parent_name,
                    frames[frame][parent_name][1],
                )
            )

        with torch.no_grad():
            smplx_output = body_model(
                global_orient=torch.tensor(
                    root_orient.as_rotvec().reshape(1, 1, 3), device=device
                ).float(),
                body_pose=body_pose.reshape(1, body_model.NUM_BODY_JOINTS, 3),
                return_full_pose=True,
            )

        lafan_positions = torch.from_numpy(
            np.array(
                [frames[frame][name][0] for name in frames[frame] if "Mod" not in name],
                dtype=np.float32,
            )
        ).to(device)
        lafan_centroid = lafan_positions.mean(dim=0)

        # Again, we only care about root + 21 joints
        smplx_centroid = smplx_output.joints[0, :22].mean(dim=0)

        trans_tensor[frame] = lafan_centroid - smplx_centroid
        root_orient_tensor[frame] = torch.tensor(
            root_orient.as_rotvec(), device=device
        ).float()
        body_pose_tensor[frame] = body_pose.clone()

        if args.rerun:
            rr.set_time_sequence("frame", frame)
            rr.log(
                "nameless_motion",
                rr.Mesh3D(
                    vertex_positions=(trans_tensor[frame] + smplx_output.vertices[0])
                    .detach()
                    .cpu()
                    .numpy(),
                    triangle_indices=body_model.faces,
                ),
            )

    if args.output_file is not None:
        full_poses_tensor = torch.cat(
            [root_orient_tensor.unsqueeze(1), body_pose_tensor],
            dim=1,
        )
        if args.model_type == "smplx":
            hand_pose_tensor = torch.zeros(
                (n_frames, 3 + 15 * 2, 3), device=device, dtype=torch.float32
            )
            full_poses_tensor = torch.cat([full_poses_tensor, hand_pose_tensor], dim=1)
        np.savez(
            args.output_file,
            trans=trans_tensor.cpu().numpy(),
            gender="neutral",
            mocap_frame_rate=30,
            betas=body_model.betas.detach().cpu().squeeze().numpy(),
            poses=full_poses_tensor.cpu().numpy(),
        )
