""" 本コードはサンプルで、PyTorch の学習率減衰の使い方を示す。
"""

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
