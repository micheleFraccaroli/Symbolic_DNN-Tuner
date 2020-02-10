action(reg_l2) :- ove.
action(inc_dropout) :- \+ove.

action(decr_lr) :- \+und.
action(inc_neurons) :- und.

action(decr_lr) :- True.

action(inc_batch_size) :- flo.
action(decr_lr) :- \+flo.
