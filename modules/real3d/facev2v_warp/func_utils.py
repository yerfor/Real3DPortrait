import torch
import torch.nn.functional as F


def rotation_matrix_x(theta):
    theta = theta.reshape(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([c, z, s], 2),
            torch.cat([z, o, z], 2),
            torch.cat([-s, z, c], 2),
        ],
        1,
    )


def rotation_matrix_y(theta):
    theta = theta.reshape(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([o, z, z], 2),
            torch.cat([z, c, -s], 2),
            torch.cat([z, s, c], 2),
        ],
        1,
    )


def rotation_matrix_z(theta):
    theta = theta.reshape(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([c, -s, z], 2),
            torch.cat([s, c, z], 2),
            torch.cat([z, z, o], 2),
        ],
        1,
    )


def transform_kp(canonical_kp, yaw, pitch, roll, t, delta):
    # [N,K,3] [N,] [N,] [N,] [N,3] [N,K,3]
    # y, x, z
    # w, h, d
    rot_mat = rotation_matrix_y(pitch) @ rotation_matrix_x(yaw) @ rotation_matrix_z(roll)
    transformed_kp = torch.matmul(rot_mat.unsqueeze(1), canonical_kp.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(1) + delta
    return transformed_kp, rot_mat


def transform_kp_with_new_pose(canonical_kp, yaw, pitch, roll, t, delta, new_yaw, new_pitch, new_roll):
    # [N,K,3] [N,] [N,] [N,] [N,3] [N,K,3]
    # y, x, z
    # w, h, d
    old_rot_mat = rotation_matrix_y(pitch) @ rotation_matrix_x(yaw) @ rotation_matrix_z(roll)
    rot_mat = rotation_matrix_y(new_pitch) @ rotation_matrix_x(new_yaw) @ rotation_matrix_z(new_roll)
    R = torch.matmul(rot_mat, torch.inverse(old_rot_mat))
    transformed_kp = (
        torch.matmul(rot_mat.unsqueeze(1), canonical_kp.unsqueeze(-1)).squeeze(-1)
        + t.unsqueeze(1)
        + torch.matmul(R.unsqueeze(1), delta.unsqueeze(-1)).squeeze(-1)
    )
    zt = 0.33 - transformed_kp[:, :, 2].mean()
    transformed_kp = transformed_kp + torch.FloatTensor([0, 0, zt]).cuda()
    return transformed_kp, rot_mat


def make_coordinate_grid_2d(spatial_size):
    h, w = spatial_size
    x = torch.arange(h).cuda()
    y = torch.arange(w).cuda()
    x = 2 * (x / (h - 1)) - 1
    y = 2 * (y / (w - 1)) - 1
    xx = x.reshape(-1, 1).repeat(1, w)
    yy = y.reshape(1, -1).repeat(h, 1)
    meshed = torch.cat([yy.unsqueeze(2), xx.unsqueeze(2)], 2)
    return meshed


def make_coordinate_grid_3d(spatial_size):
    d, h, w = spatial_size
    z = torch.arange(d).cuda()
    x = torch.arange(h).cuda()
    y = torch.arange(w).cuda()
    z = 2 * (z / (d - 1)) - 1
    x = 2 * (x / (h - 1)) - 1
    y = 2 * (y / (w - 1)) - 1
    zz = z.reshape(-1, 1, 1).repeat(1, h, w)
    xx = x.reshape(1, -1, 1).repeat(d, 1, w)
    yy = y.reshape(1, 1, -1).repeat(d, h, 1)
    meshed = torch.cat([yy.unsqueeze(3), xx.unsqueeze(3), zz.unsqueeze(3)], 3)
    return meshed


def out2heatmap(out, temperature=0.1):
    final_shape = out.shape
    heatmap = out.reshape(final_shape[0], final_shape[1], -1)
    heatmap = F.softmax(heatmap / temperature, dim=2)
    heatmap = heatmap.reshape(*final_shape)
    return heatmap


def heatmap2kp(heatmap):
    shape = heatmap.shape
    grid = make_coordinate_grid_3d(shape[2:]).unsqueeze(0).unsqueeze(0)
    kp = (heatmap.unsqueeze(-1) * grid).sum(dim=(2, 3, 4))
    return kp


def kp2gaussian_2d(kp, spatial_size, kp_variance=0.01):
    N, K = kp.shape[:2]
    coordinate_grid = make_coordinate_grid_2d(spatial_size).reshape(1, 1, *spatial_size, 2).repeat(N, K, 1, 1, 1)
    mean = kp.reshape(N, K, 1, 1, 2)
    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def kp2gaussian_3d(kp, spatial_size, kp_variance=0.01):
    N, K = kp.shape[:2]
    coordinate_grid = make_coordinate_grid_3d(spatial_size).reshape(1, 1, *spatial_size, 3).repeat(N, K, 1, 1, 1, 1)
    mean = kp.reshape(N, K, 1, 1, 1, 3)
    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def create_heatmap_representations(fs, kp_s, kp_d):
    spatial_size = fs.shape[2:]
    heatmap_d = kp2gaussian_3d(kp_d, spatial_size)
    heatmap_s = kp2gaussian_3d(kp_s, spatial_size)
    heatmap = heatmap_d - heatmap_s
    zeros = torch.zeros(heatmap.shape[0], 1, *spatial_size).cuda()
    # [N,21,16,64,64]
    heatmap = torch.cat([zeros, heatmap], dim=1)
    # [N,21,1,16,64,64]
    heatmap = heatmap.unsqueeze(2)
    return heatmap


def create_sparse_motions(fs, kp_s, kp_d, Rs, Rd):
    N, _, D, H, W = fs.shape
    K = kp_s.shape[1]
    identity_grid = make_coordinate_grid_3d((D, H, W)).reshape(1, 1, D, H, W, 3).repeat(N, 1, 1, 1, 1, 1)
    # [N,20,16,64,64,3]
    coordinate_grid = identity_grid.repeat(1, K, 1, 1, 1, 1) - kp_d.reshape(N, K, 1, 1, 1, 3)
    # [N,1,1,1,1,3,3]
    jacobian = torch.matmul(Rs, torch.inverse(Rd)).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
    coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1)).squeeze(-1)
    driving_to_source = coordinate_grid + kp_s.reshape(N, K, 1, 1, 1, 3)
    sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
    # sparse_motions = driving_to_source
    # [N,21,16,64,64,3]
    return sparse_motions

def create_deformed_source_image2d(fs, sparse_motions):
    N, _, H, W = fs.shape
    K = sparse_motions.shape[1] - 1
    # [N*21,4,16,64,64]
    source_repeat = fs.unsqueeze(1).repeat(1, K + 1, 1, 1, 1).reshape(N * (K + 1), -1, H, W)
    # [N*21,16,64,64,3]
    sparse_motions = sparse_motions.reshape((N * (K + 1), H, W, -1))
    # [N*21,4,16,64,64]
    sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True)
    sparse_deformed = sparse_deformed.reshape((N, K + 1, -1, H, W))
    # [N,21,4,16,64,64]
    return sparse_deformed

def create_deformed_source_image(fs, sparse_motions):
    N, _, D, H, W = fs.shape
    K = sparse_motions.shape[1] - 1
    # [N*21,4,16,64,64]
    source_repeat = fs.unsqueeze(1).repeat(1, K + 1, 1, 1, 1, 1).reshape(N * (K + 1), -1, D, H, W)
    # [N*21,16,64,64,3]
    sparse_motions = sparse_motions.reshape((N * (K + 1), D, H, W, -1))
    # [N*21,4,16,64,64]
    sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True)
    sparse_deformed = sparse_deformed.reshape((N, K + 1, -1, D, H, W))
    # [N,21,4,16,64,64]
    return sparse_deformed


def apply_imagenet_normalization(input):
    mean = input.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    output = (input - mean) / std
    return output


def apply_vggface_normalization(input):
    mean = input.new_tensor([129.186279296875, 104.76238250732422, 93.59396362304688]).reshape(1, 3, 1, 1)
    std = input.new_tensor([1, 1, 1]).reshape(1, 3, 1, 1)
    output = (input * 255 - mean) / std
    return output
