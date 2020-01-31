/**
gap_tr_te_acc.
gap_tr_te_loss.
low_acc.
high_loss.
growing_loss_trend.
up_down_loss.
**/

problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.

query(problem(_))