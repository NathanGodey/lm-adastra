from torch.optim.lr_scheduler import LambdaLR


def get_linear_with_min_lr_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_rate):
  if warmup_steps > total_steps:
    raise ValueError(f"Provided larger warmup ({warmup_steps}) than total steps ({total_steps}) in LR scheduler.")
  assert 0 < min_lr_rate <= 1

  def _linear_sched_func(i):
    if i < warmup_steps:
      return i/warmup_steps
    else:
      step_ratio = 1 - (i - warmup_steps) / (total_steps - warmup_steps)
      return step_ratio + (1-step_ratio) * min_lr_rate

  return LambdaLR(optimizer, _linear_sched_func)