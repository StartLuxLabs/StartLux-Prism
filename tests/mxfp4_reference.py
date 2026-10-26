"""CPU-only MXFP4 reference: nearest representable value, ties to even code."""
import torch


def quantize(x, scaling='ceil', rotate=False):
    x = x.cpu().double().reshape(x.shape[0], -1, 32)
    if rotate:
        h = torch.ones((1, 1), dtype=torch.float64)
        for _ in range(5):
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        x = x @ (h * 0.1767766952966369)
    bad = ~torch.isfinite(x).all(-1)
    finite = torch.where(torch.isfinite(x), x, 0.)
    mx = finite.abs().amax(-1)
    e = torch.ceil(torch.log2((mx / 6.).clamp_min(2.**-126))).clamp(-126, 127)
    e[mx == 0] = 0
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    def nearest(exp):
        v = finite / torch.exp2(exp)[..., None]
        distance = (v.abs()[..., None] - levels).abs()
        ties = distance == distance.amin(-1, keepdim=True)
        codes = torch.arange(8)
        priority = torch.where(codes % 2 == 0, codes, codes + 8)
        choice = torch.where(ties, priority, 100).argmin(-1).to(torch.uint8)
        error = ((v.abs() - levels[choice.long()]) * torch.exp2(exp)[..., None]).square().sum(-1)
        return choice, error
    choice, err_hi = nearest(e)
    if scaling == 'mse':
        lower, err_lo = nearest(e - 1)
        use_lower = (err_lo < err_hi) & (e > -126)
        choice = torch.where(use_lower[..., None], lower, choice)
        e = torch.where(use_lower, e - 1, e)
    v = finite / torch.exp2(e)[..., None]
    choice |= torch.signbit(v).to(torch.uint8) * 8
    packed = choice[..., ::2] | (choice[..., 1::2] << 4)
    packed[bad] = 0
    scale = (e + 127).to(torch.uint8)
    scale[bad] = 255
    return packed.flatten(1), scale


def dequantize(q, s):
    q, s = q.cpu(), s.cpu()
    c = torch.stack((q & 15, q >> 4), -1).flatten(1)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    v = levels[(c & 7).long()] * torch.where((c & 8) != 0, -1., 1.)
    scale = torch.exp2(s.double() - 127)
    scale[s == 255] = float('nan')
    return (v.reshape(q.shape[0], -1, 32) * scale[..., None]).flatten(1)
