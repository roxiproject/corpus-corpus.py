"""Learning rate schedules: linear warmup and cosine decay.

Both are expressed as pure functions of step index so they're trivially
testable at specific points.
"""

import math


def linear_warmup(step, warmup_steps, base_lr):
    """Linearly ramp from 0 to base_lr over `warmup_steps` steps.

    step is 0-indexed. At step 0 the lr is 0 (unless warmup_steps == 0,
    in which case base_lr is returned immediately). At step ==
    warmup_steps - 1 the lr is (warmup_steps-1)/warmup_steps * base_lr;
    it reaches exactly base_lr once step >= warmup_steps.
    """
    if warmup_steps <= 0:
        return base_lr
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step / warmup_steps)


def cosine_decay(step, total_steps, base_lr, min_lr=0.0):
    """Cosine decay from base_lr down to min_lr over total_steps steps."""
    if total_steps <= 0:
        return base_lr
    step = min(step, total_steps)
    progress = step / total_steps
    cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_term


def warmup_then_cosine(step, warmup_steps, total_steps, base_lr, min_lr=0.0):
    """Linear warmup for `warmup_steps`, then cosine decay for the remainder."""
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps, base_lr)
    remaining_total = max(total_steps - warmup_steps, 1)
    remaining_step = step - warmup_steps
    return cosine_decay(remaining_step, remaining_total, base_lr, min_lr)
