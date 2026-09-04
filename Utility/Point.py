import torch
import pypose as pp

# Camera-axis convention. Internally MAC-VO uses NED camera axes [forward, right, down]
# (see pixel2point_NED below); several datasets ship ground truth in OpenCV/EDN
# [right, down, forward]. EDN2NED converts EDN coordinates into NED; NED2EDN is its inverse
# and is what you right-multiply onto a camera->world pose to rebase its BODY axes into NED.
EDN2NED = pp.from_matrix(torch.tensor([
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
]), pp.SE3_type)
NED2EDN = EDN2NED.Inv()

def filterPointsInRange(pts1:torch.Tensor, u_range: tuple[int, int], v_range: tuple[int, int]) -> torch.Tensor:
    u_min, u_max = u_range
    v_min, v_max = v_range

    u_selector = torch.logical_and(pts1[..., 0] < u_max, pts1[..., 0] > u_min)
    v_selector = torch.logical_and(pts1[..., 1] < v_max, pts1[..., 1] > v_min)
    selector = torch.logical_and(u_selector, v_selector)

    return selector

def pixel2point_NED(pixels: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor):
    # pp.pixel2point will output points in EDN coordinate, we will convert it to NED coord.
    return pp.pixel2point(pixels, depths, intrinsics).roll(shifts=1, dims=-1)

def point2pixel_NED(points: torch.Tensor, intrinsics: torch.Tensor):
    # pp.pixel2point will output points in EDN coordinate, we will convert it to NED coord.
    return pp.point2pixel(points.roll(shifts=-1, dims=-1), intrinsics)
