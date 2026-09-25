import math
import torch
import torch.nn.init as init

src = (
    'data/weights/'
    'hybrid_iros_nds7011_lgpc_iter81000.pth'
)

dst = (
    'data/weights/'
    'hybrid_iros7011_lgpc81000_rrrf_pred_reset.pth'
)

ckpt = torch.load(src, map_location='cpu')
sd = ckpt['state_dict']

prefix = 'bbox_head.calibration_predictor.'

# First FC: fresh initialization
w1 = sd[prefix + '0.weight']
b1 = sd[prefix + '0.bias']

init.kaiming_uniform_(w1, a=math.sqrt(5))

fan_in, _ = init._calculate_fan_in_and_fan_out(w1)
bound = 1 / math.sqrt(fan_in)
init.uniform_(b1, -bound, bound)

# Final FC: identity residual = zero
sd[prefix + '2.weight'].zero_()
sd[prefix + '2.bias'].zero_()

torch.save(ckpt, dst)

print('[PASS] New RRRF predictor initialized')
print(dst)