:- use_module(library(lists)).

% DIAGNOSIS SECTION ---------------------------------------------------------------------------------------------------

% utility
abs2(X,Y) :- Y is abs(X).
isclose(X,Y,W) :- D is X - Y, abs2(D,D1), D1 =< W.
add_to_UpList([_],0).
add_to_UpList([H|[H1|T]], U) :- add_to_UpList([H1|T], U1), H < H1, U is U1+1.
add_to_DownList([_],0).
add_to_DownList([H|[H1|T]], U) :- add_to_UpList([H1|T], U1), H > H1, U is U1+1.

% analysis
gap_tr_te_acc :- a(A), va(VA), last(A,LTA), last(VA,ScoreA),
                Res is LTA - ScoreA, abs2(Res,Res1), Res1 > 0.2.
gap_tr_te_loss :- l(L), vl(VL), last(L,LTL), last(VL,ScoreL),
                Res is LTL - ScoreL, abs2(Res,Res1), Res1 > 0.2.
low_acc :- a(A), last(A,LTA),
                Res is LTA - 1.0, abs2(Res,Res1), Res1 > 0.35.
high_loss :- l(L), last(L,LTL), isclose(LTL,0,1.7).
growing_loss_trend :- add_to_UpList(sl,usl), length(usl,length_u), G is (usl*100)/length_u, G < 50.
up_down_loss :- add_to_UpList(sl,usl), add_to_DownList(sl,dsl), isclose(usl,dsl,150), usl > 0, dsl > 0.

% problems
problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.


% TUNING SECTION ------------------------------------------------------------------------------------------------------

% 0.8::ove.
% 0.7::und.
% 0.85::flo.

action(reg_l2, overfitting) :- problem(overfitting), ove.
action(inc_dropout, overfitting) :- problem(overfitting), \+ove.

action(decr_lr, underfitting) :- problem(underfitting), \+und.
action(inc_neurons, underfitting) :- problem(underfitting), und.

action(decr_lr, inc_loss) :- problem(inc_loss).

action(inc_batch_size, floating_loss) :- problem(floating_loss), flo.
action(decr_lr, floating_loss) :- problem(floating_loss), \+flo.

query(action(_,_)).