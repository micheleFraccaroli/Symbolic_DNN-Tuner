/**
overfitting.
underfitting.
inc_loss.
floating_loss.
**/

0.8::action(reg_l2) :- overfitting.
0.2::action(inc_dropout) :- overfitting.

0.3::action(decr_lr) :- underfitting.
0.7::action(inc_neurons) :- underfitting.

action(decr_lr) :- inc_loss.

0.85::action(inc_batch_size) :- floating_loss.
0.15::action(decr_lr) :- floating_loss.

query(action(_)).