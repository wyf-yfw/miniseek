"""Max Steps：新开与续训都是本轮预算；不把 ckpt 旧步数并进 LR/进度分母。"""
from algo.train_pretrain import lr_at_step, resolve_step_budget


def test_fresh_max_steps_is_absolute():
    total, limit = resolve_step_budget(0, 1000, plan_steps=94849)
    assert total == 1000
    assert limit == 1000


def test_resume_max_steps_is_this_run_only():
    total, limit = resolve_step_budget(106586, 1000, plan_steps=94849)
    assert total == 1000
    assert limit == 1000


def test_resume_without_max_steps_uses_plan_only():
    total, limit = resolve_step_budget(106586, None, plan_steps=94849)
    assert total == 94849
    assert limit is None


def test_fresh_without_max_steps_uses_plan():
    total, limit = resolve_step_budget(0, None, plan_steps=20)
    assert total == 20
    assert limit is None


def test_non_positive_max_steps_means_no_cap():
    total, limit = resolve_step_budget(106586, 0, plan_steps=10)
    assert total == 10
    assert limit is None


def test_resume_lr_starts_near_peak_not_min_lr():
    """续训 lifetime=106586、本轮 1000、warmup=200 → 本轮 step1 在 warmup 爬坡，绝不是 ≈min_lr。"""
    resumed_step = 106586
    max_steps = 1000
    lr = 1e-3
    min_lr = 1e-5
    warmup = 200
    schedule_total, _ = resolve_step_budget(resumed_step, max_steps, plan_steps=94849)
    assert schedule_total == 1000
    # 本轮 step=1（训练循环里 next_step=global_step+1）
    at_1 = lr_at_step(1, base_lr=lr, min_lr=min_lr, warmup_steps=warmup, total_steps=schedule_total)
    at_warm = lr_at_step(warmup, base_lr=lr, min_lr=min_lr, warmup_steps=warmup, total_steps=schedule_total)
    assert abs(at_1 - lr * (1.0 / warmup)) < 1e-12
    assert abs(at_warm - lr) < 1e-12
    # 若误用旧公式 lifetime/(lifetime+extra) 会贴在 min_lr
    wrong_total = resumed_step + max_steps
    wrong = lr_at_step(
        resumed_step + 1,
        base_lr=lr,
        min_lr=min_lr,
        warmup_steps=warmup,
        total_steps=wrong_total,
    )
    assert wrong < min_lr * 1.5
    assert at_warm > wrong * 50
