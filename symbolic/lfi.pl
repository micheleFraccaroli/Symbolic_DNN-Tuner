t(0.8)::ove.
t(0.7)::und.
t(0.85)::flo.

action(reg_l2, overfitting) :- ove.
action(inc_dropout, overfitting) :- \+ove.

action(decr_lr, underfitting) :- \+und.
action(inc_neurons, underfitting) :- und.

action(decr_lr, inc_loss) :- true.

action(inc_batch_size, floating_loss) :- flo.
action(decr_lr, floating_loss) :- \+flo.

