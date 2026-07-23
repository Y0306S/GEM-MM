from gem_mm.entropy import fork_entropy_reward, group_normalize, token_entropy

def test_token_entropy_uniform():
    h = token_entropy([0.25, 0.25, 0.25, 0.25])
    assert 1.38 < h < 1.40

def test_reward_and_adv():
    r = fork_entropy_reward([0.1, 2.0, 0.2], lambda_weight=2.0, top_m_ratio=0.5)
    assert r > 0
    adv = group_normalize([1.0, 2.0, 3.0])
    assert abs(sum(adv)) < 1e-6
