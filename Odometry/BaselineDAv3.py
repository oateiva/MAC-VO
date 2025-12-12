from types import SimpleNamespace
import torch
import roma
import pypose as pp

from typing import Any
# CI/CD PyRight pass cannot cover the DPVO as it requires to compile CUDA kernels
from Module.Network.Depth.DepthAnythingV3.api import DepthAnything3

from DataLoader import SequenceBase, Frame
from Module.Map import VisualMap, FrameNode

from .Interface import IOdometry
import torchvision.transforms as T


class DepthAnythingV3(IOdometry[Frame]):
    def __init__(self, weight_file: str = "", **kwargs) -> None:
        super().__init__()
        self.weight_file = weight_file

        self.map  = VisualMap()
        self.dav3 = DepthAnything3.from_pretrained(self.weight_file)
        self.dav3.to(torch.device("cuda"))

        self.Ks, self.poses, self.timestep = [], None, None
        self.Ts    = []
        self.T_BSs = []
        self.absolute_poses = [None] * 10000  # Pre-allocate for up to 10k frames

    @classmethod
    def from_config(cls: type["DepthAnythingV3"], cfg: SimpleNamespace, seq: SequenceBase[Frame]) -> "DeepPatchVO":
        sample_frame = seq[0]
        return cls(**vars(cfg.Odometry.args), height=sample_frame.camera.height, width=sample_frame.camera.width)

    @torch.no_grad()
    @torch.inference_mode()
    def run(self, frame: Frame) -> None:
        if len(frame.idx) == 1:
            raise AssertionError("window length must be greater than 1 for odometry estimation")
        self.Ks.append(frame.camera.K)
        self.T_BSs.append(frame.camera.T_BS)
        self.Ts.append(frame.camera.frame_ns)
        image = frame.camera.imageL
        intrinsics = frame.camera.K
        image, extrinsics, intrinsics = self.dav3._prepare_model_inputs(
            imgs_cpu=image,
            extrinsics=None,
            intrinsics=intrinsics,
        )
        image = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        prediction = self.dav3.forward(
            image=image,
            extrinsics=None,
            intrinsics=intrinsics,
            export_feat_layers=[],
            infer_gs=False, # No need for Gaussian branch in standard depth inference
        )
        extrinsics_SE3 = prediction.extrinsics

        H = torch.eye(4, device=extrinsics_SE3.device)
        H = H.unsqueeze(0).repeat(extrinsics_SE3.size(1), 1, 1).unsqueeze(0)
        H[:, :, :3, :4] = extrinsics_SE3

        if frame.idx[0] == 0 :
            for i in range(frame.idx[0],frame.idx[-1]+1):
                pp_from_matrix = pp.from_matrix(H[:, i, :, :], ltype=pp.SE3_type)
                # pp_SE3 = pp.SE3(pp_from_matrix)
                pp_tensor = pp_from_matrix.tensor()
                self.absolute_poses[i] = pp_tensor
        else:
            prev_pose = self.absolute_poses[frame.idx[0]]
            j=0
            for i in range(frame.idx[0],frame.idx[-1]+1):
                ##### OPTION 1
                prev_pose_matrix = pp.SE3(prev_pose).matrix()
                current_pose_matrix = H[:, j, :, :]
                curr_pose = prev_pose_matrix @ current_pose_matrix
                pp_from_matrix = pp.from_matrix(curr_pose, ltype=pp.SE3_type)
                pp_tensor = pp_from_matrix.tensor()
                self.absolute_poses[i] = pp_tensor
                ##### OPTION 2
                # if j==0:
                #     prev_pose = prev_pose
                # else:
                #     prev_pose = pp.from_matrix(H[:, j-1, :, :], ltype=pp.SE3_type).tensor()

                # prev_pose_matrix = pp.SE3(prev_pose).matrix()
                # current_pose_matrix = H[:, j, :, :]
                # curr_pose = prev_pose_matrix @ current_pose_matrix
                # pp_from_matrix = pp.from_matrix(curr_pose, ltype=pp.SE3_type)
                # pp_tensor = pp_from_matrix.tensor()
                # self.absolute_poses[i] = pp_tensor

                j+=1


        torch.cuda.empty_cache()

    def get_map(self) -> VisualMap:
        return self.map

    @torch.no_grad()
    @torch.inference_mode()
    def terminate(self) -> None:
        super().terminate()
        # self.poses = self.poses[..., [2, 0, 1, 5, 3, 4, 6]]
        # self.poses = pp.SE3(self.poses)

        pose = torch.stack([p for p in self.absolute_poses if p is not None]) # 20 1 7
        T_BSs = torch.stack(self.T_BSs).squeeze(1) # 20 1 7
        Ks    = torch.cat(self.Ks, dim=0) # 20 3 3
        time_ns    = torch.tensor(self.Ts, dtype=torch.long) # 20

        n_frame = T_BSs.size(0)
        need_interp = torch.zeros((n_frame,), dtype=torch.bool) # 20
        pose = pose[:n_frame].squeeze(1) # 20 1 7
        baseline = torch.zeros((n_frame,), dtype=torch.float32) # 20

        framenode = FrameNode.init(
                {
                "pose": pose,
                "T_BS": T_BSs,
                "K"   : Ks,
                "need_interp": need_interp,
                "time_ns"    : time_ns,
                "baseline": baseline,
                }
            )
        self.map.frames.push(
            framenode
        )
