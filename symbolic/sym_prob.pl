0.99::eve.
action(reg_l2,overfitting) :- eve.
action(decr_lr,inc_loss) :- eve.
0.4::action(inc_dropout,overfitting).
0.6::action(data_augmentation,overfitting).
0.3::action(decr_lr,underfitting).
0.7::action(inc_neurons,underfitting).
0.85::action(inc_batch_size,floating_loss).
0.15::action(decr_lr,floating_loss).
