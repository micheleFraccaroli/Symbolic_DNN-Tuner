l([2483.6246006804467, 2.308219975376129, 2.305401832675934, 2.3035286644935606, 2.303716730213165, 2.3035660720825195, 2.304299839115143, 2.3034860894203186, 2.3041674651145936, 2.3041675618171693]).
sl([2483.6246006804467, 1491.0980483984183, 895.5809897721214, 538.2700053290702, 323.8834898895273, 195.25152036254937, 118.07263215317568, 71.76497372767352, 43.980651222649946, 27.310057758316837]).
a([0.10289999842643738, 0.10349999964237214, 0.10233999848365784, 0.10256399846076966, 0.10145839831829072, 0.10071503985881805, 0.09990902464962005, 0.09938541484699248, 0.09879125018611908, 0.10007474934873199]).
sa([0.1029, 0.1044, 0.1006, 0.1029, 0.0998, 0.0996, 0.0987, 0.0986, 0.0979, 0.102]).
vl([2.3114402429262797, 2.303875916004181, 2.303608958562215, 2.303401670773824, 2.303127555370331, 2.3037121540705363, 2.304613801956177, 2.3031674184799193, 2.30406814289093, 2.304047981421153]).
va([0.096666664, 0.09933333, 0.096666664, 0.09933333, 0.1005, 0.10016666, 0.10016666, 0.100833334, 0.102, 0.1005]).
int_loss(8.7373).
int_slope(40.5).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
action(decr_lr,high_lr) :- eve, problem(high_lr).
action(incr_lr,low_lr) :- eve, problem(low_lr).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.65::action(inc_neurons,underfitting):- problem(underfitting).
0.75::action(new_fc_layer,underfitting):- problem(underfitting).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).

% DIAGNOSIS SECTION ----------------------------------------------------------------------------------------------------
:- use_module(library(lists)).

% UTILITY
abs2(X,Y) :- Y is abs(X).
isclose(X,Y,W) :- D is X - Y, abs2(D,D1), D1 =< W.

add_to_UpList([_],0).
add_to_UpList([H|[H1|T]], U) :- add_to_UpList([H1|T], U1), H =< H1, U is U1+1.
add_to_UpList([H|[H1|T]], U) :- add_to_UpList([H1|T], U1), H > H1, U is U1+0.

add_to_DownList([_],0).
add_to_DownList([H|[H1|T]], U) :- add_to_DownList([H1|T], U1), H > H1, U is U1+1.
add_to_DownList([H|[H1|T]], U) :- add_to_DownList([H1|T], U1), H =< H1, U is U1+0.

area_sub(R) :- int_loss(A), int_slope(B), Rt is A - B, abs2(Rt,R).
threshold_up(Th) :- int_slope(A), Th is A/4.
threshold_down(Th) :- int_slope(A), Th is A*(3/4).

% ANALYSIS
gap_tr_te_acc :- a(A), va(VA), last(A,LTA), last(VA,ScoreA),
                Res is LTA - ScoreA, abs2(Res,Res1), Res1 > 0.2.
gap_tr_te_loss :- l(L), vl(VL), last(L,LTL), last(VL,ScoreL),
                Res is LTL - ScoreL, abs2(Res,Res1), Res1 > 0.2.
low_acc :- a(A), last(A,LTA),
                Res is LTA - 1.0, abs2(Res,Res1), Res1 > 0.37.
high_loss :- l(L), last(L,LTL), isclose(LTL,0,1.7).
growing_loss_trend :- l(L),add_to_UpList(L,Usl), length(L,Length_u), G is (Usl*100)/Length_u, G > 50.
up_down_acc :- a(A),add_to_UpList(A,Usa), add_to_DownList(A,Dsa), isclose(Usa,Dsa,150), Usa > 0, Dsa > 0.
up_down_loss :- l(L),add_to_UpList(L,Usl), add_to_DownList(L,Dsl), isclose(Usl,Dsl,150), Usl > 0, Dsl > 0.
to_low_lr :- area_sub(As), threshold_up(Th), As < Th.
to_high_lr :- area_sub(As), threshold_down(Th), As > Th.

% POSSIBLE PROBLEMS
problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.
problem(low_lr) :- to_low_lr.
problem(high_lr) :- to_high_lr.

% QUERY ----------------------------------------------------------------------------------------------------------------
query(action(_,_)).