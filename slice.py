import torch

def lerp_weight(x, xs):
  """Linear interpolation weight from a sample at x to xs.
  Returns the linear interpolation weight of a "query point" at coordinate `x`
  with respect to a "sample" at coordinate `xs`.
  The integer coordinates `x` are at pixel centers.
  The floating point coordinates `xs` are at pixel edges.
  (OpenGL convention).
  Args:
    x: "Query" point position.
    xs: "Sample" position.
  Returns:
    - 1 when x = xs.
    - 0 when |x - xs| > 1.
  """
  dx = x - xs
  abs_dx = abs(dx)
  return torch.maximum(torch.tensor(1.0).to(x.device) - abs_dx, torch.tensor(0.0).to(x.device))


def smoothed_abs(x, eps):
  """A smoothed version of |x| with improved numerical stability."""
  return torch.sqrt(torch.multiply(x, x) + eps)


def smoothed_lerp_weight(x, xs):
  """Smoothed version of `LerpWeight` with gradients more suitable for backprop.
  Let f(x, xs) = LerpWeight(x, xs)
               = max(1 - |x - xs|, 0)
               = max(1 - |dx|, 0)
  f is not smooth when:
  - |dx| is close to 0. We smooth this by replacing |dx| with
    SmoothedAbs(dx, eps) = sqrt(dx * dx + eps), which has derivative
    dx / sqrt(dx * dx + eps).
  - |dx| = 1. When smoothed, this happens when dx = sqrt(1 - eps). Like ReLU,
    We just ignore this (in the implementation below, when the floats are
    exactly equal, we choose the SmoothedAbsGrad path since it is more useful
    than returning a 0 gradient).
  Args:
    x: "Query" point position.
    xs: "Sample" position.
    eps: a small number.
  Returns:
    max(1 - |dx|, 0) where |dx| is smoothed_abs(dx).
  """
  eps = torch.tensor(1e-8).to(torch.float32).to(x.device)
  dx = x - xs
  abs_dx = smoothed_abs(dx, eps)
  return torch.maximum(torch.tensor(1.0).to(x.device) - abs_dx, torch.tensor(0.0).to(x.device))

def _bilateral_slice(grid, guide):
    """Slices a bilateral grid using the a guide image.
    Args:
      grid: The bilateral grid with shape (gh, gw, gd, gc).
      guide: A guide image with shape (h, w). Values must be in the range [0, 1].
    Returns:
      sliced: An image with shape (h, w, gc), computed by trilinearly
      interpolating for each grid channel c the grid at 3D position
      [(i + 0.5) * gh / h,
       (j + 0.5) * gw / w,
       guide(i, j) * gd]
    """
    dev = grid.device
    ii, jj = torch.meshgrid(
        [torch.arange(guide.shape[0]).to(dev), torch.arange(guide.shape[1]).to(dev)], indexing='ij')

    scale_i = grid.shape[0] / guide.shape[0]
    scale_j = grid.shape[1] / guide.shape[1]

    gif = (ii + 0.5) * scale_i
    gjf = (jj + 0.5) * scale_j
    gkf = guide * grid.shape[2]

    # Compute trilinear interpolation weights without clamping.
    gi0 = torch.floor(gif - 0.5).to(torch.int32)
    gj0 = torch.floor(gjf - 0.5).to(torch.int32)
    gk0 = torch.floor(gkf - 0.5).to(torch.int32)
    gi1 = gi0 + 1
    gj1 = gj0 + 1
    gk1 = gk0 + 1

    wi0 = lerp_weight(gi0 + 0.5, gif)
    wi1 = lerp_weight(gi1 + 0.5, gif)
    wj0 = lerp_weight(gj0 + 0.5, gjf)
    wj1 = lerp_weight(gj1 + 0.5, gjf)
    wk0 = smoothed_lerp_weight(gk0 + 0.5, gkf)
    wk1 = smoothed_lerp_weight(gk1 + 0.5, gkf)

    w_000 = wi0 * wj0 * wk0
    w_001 = wi0 * wj0 * wk1
    w_010 = wi0 * wj1 * wk0
    w_011 = wi0 * wj1 * wk1
    w_100 = wi1 * wj0 * wk0
    w_101 = wi1 * wj0 * wk1
    w_110 = wi1 * wj1 * wk0
    w_111 = wi1 * wj1 * wk1

    # But clip when indexing into `grid`.
    gi0c = gi0.clip(0, grid.shape[0] - 1).to(torch.long)
    gj0c = gj0.clip(0, grid.shape[1] - 1).to(torch.long)
    gk0c = gk0.clip(0, grid.shape[2] - 1).to(torch.long)

    gi1c = (gi0 + 1).clip(0, grid.shape[0] - 1).to(torch.long)
    gj1c = (gj0 + 1).clip(0, grid.shape[1] - 1).to(torch.long)
    gk1c = (gk0 + 1).clip(0, grid.shape[2] - 1).to(torch.long)

    #        ijk: 0 means floor, 1 means ceil.
    grid_val_000 = grid[gi0c, gj0c, gk0c, :]
    grid_val_001 = grid[gi0c, gj0c, gk1c, :]
    grid_val_010 = grid[gi0c, gj1c, gk0c, :]
    grid_val_011 = grid[gi0c, gj1c, gk1c, :]
    grid_val_100 = grid[gi1c, gj0c, gk0c, :]
    grid_val_101 = grid[gi1c, gj0c, gk1c, :]
    grid_val_110 = grid[gi1c, gj1c, gk0c, :]
    grid_val_111 = grid[gi1c, gj1c, gk1c, :]

    # Append a singleton "channels" dimension.
    w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111 = torch.atleast_3d(
        w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111)

    # TODO(jiawen): Cache intermediates and pass them in.
    # Just pass out w_ijk and the same ones multiplied by by dwk.
    return (torch.multiply(w_000, grid_val_000) +
            torch.multiply(w_001, grid_val_001) +
            torch.multiply(w_010, grid_val_010) +
            torch.multiply(w_011, grid_val_011) +
            torch.multiply(w_100, grid_val_100) +
            torch.multiply(w_101, grid_val_101) +
            torch.multiply(w_110, grid_val_110) +
            torch.multiply(w_111, grid_val_111))

@torch.jit.script
def batch_bilateral_slice(grid, guide):
    res = []
    for i in range(grid.shape[0]):
        res.append(_bilateral_slice(grid[i], guide[i]).unsqueeze(0))
    return torch.concat(res, 0)

def trace_bilateral_slice(grid, guide):
    return batch_bilateral_slice(grid, guide)


# grid: The bilateral grid with shape (gh, gw, gd, gc).
# guide: A guide image with shape (h, w). Values must be in the range [0, 1].

grid = torch.rand(1, 3, 3, 8, 12).cuda()
guide = torch.rand(1,16, 16).cuda()

bilateral_slice = torch.jit.trace(
    trace_bilateral_slice, (grid, guide))

bilateral_slice(grid, guide)