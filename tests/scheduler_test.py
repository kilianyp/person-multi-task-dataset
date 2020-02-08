from schedulers.scheduler import multi_step_fn
def test_multistep():
    milestones = [10, 20, 30]
    gamma = 0.1
    lr = 1
    for t in range(50):
        print(multi_step_fn(t, lr, gamma, milestones))
