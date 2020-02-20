0.99::eve.

action(reg_l2, overfitting) :- eve.
action(decr_lr, inc_loss) :- eve.

t(0.4)::action(inc_dropout, overfitting).
t(0.6)::action(data_augmentation, overfitting).

t(0.3)::action(decr_lr, underfitting).
t(0.7)::action(inc_neurons, underfitting).

t(0.85)::action(inc_batch_size, floating_loss).
t(0.15)::action(decr_lr, floating_loss).

