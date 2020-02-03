/**
* gap_tr_te_acc.
* gap_tr_te_loss.
* low_acc.
* high_loss.
* growing_loss_trend.
* up_down_loss.
* -------------------
* overfitting.
* underfitting.
* inc_loss.
* floating_loss.
**/

problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.

0.8::overf.
0.7::underf.
0.85::flo_loss.

action(reg_l2, overfitting) :- problem(overfitting), overf.
action(inc_dropout, overfitting) :- problem(overfitting), \+overf.

action(decr_lr, underfitting) :- problem(underfitting), \+underf.
action(inc_neurons, underfitting) :- problem(underfitting), underf.

action(decr_lr, inc_loss) :- problem(inc_loss).

action(inc_batch_size, floating_loss) :- problem(floating_loss), flo_loss.
action(decr_lr, floating_loss) :- problem(floating_loss), \+flo_loss.

evidence(action(_, overfitting), true) :- problem(overfitting).
evidence(action(_, underfitting), true) :- problem(underfitting).
evidence(action(_, inc_loss), true) :- problem(inc_loss).
evindence(action(_, floating_loss), true) :- problem(floating_loss).


query(action(_,_)).