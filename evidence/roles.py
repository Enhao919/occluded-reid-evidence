"""Bind auxiliary supervision to the declared role and random seed."""


def support_coefficient(arm, policy):
    allowed = {f"identity_only_seed{policy['seed']}": 0.,
               f"supported_seed{policy['seed']}": policy['support_weight']}
    if arm not in allowed:
        raise ValueError('Unknown training role or mismatched seed')
    return allowed[arm]
