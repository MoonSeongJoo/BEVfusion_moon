_base_ = ['./pgfr_p1_gate_target_5k_eval.py']

# Keep all original PGFR settings.
# Disable only the Local Proposal Fusion contribution.
model = dict(
    bbox_head=dict(
        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=0.0,
    ),
)
