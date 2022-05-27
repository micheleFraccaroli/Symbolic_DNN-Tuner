l([2.307790994644165, 2.3065333366394043, 2.305644989013672, 2.3048036098480225, 2.303957462310791, 2.3031375408172607, 2.3023061752319336, 2.3018875122070312, 2.300830125808716, 2.3004209995269775]).
sl([2.307790994644165, 2.3072879314422607, 2.3066307544708256, 2.305899896621704, 2.305122922897339, 2.3043287700653075, 2.303519732131958, 2.3028668441619873, 2.302052156820679, 2.301399693903198]).
a([0.09967999905347824, 0.09955199807882309, 0.09999520033597947, 0.10038112074136735, 0.10105267149209976, 0.1015756039738655, 0.102569363052845, 0.10354161652040482, 0.10501297051973343, 0.10627178093950271]).
sa([0.09967999905347824, 0.09935999661684036, 0.10066000372171402, 0.10096000134944916, 0.1020599976181984, 0.10236000269651413, 0.10406000167131424, 0.10499999672174454, 0.10722000151872635, 0.10815999656915665]).
vl([2.306597948074341, 2.305480480194092, 2.304462194442749, 2.303520917892456, 2.3026604652404785, 2.3018693923950195, 2.3011484146118164, 2.300464153289795, 2.2998099327087402, 2.299192190170288]).
va([0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.0997999981045723, 0.09969999641180038, 0.0997999981045723, 0.10029999911785126, 0.10090000182390213, 0.10239999741315842, 0.1054999977350235]).
int_loss(20.72231101989746).
int_slope(20.72605562210083).
lacc(0.15).
hloss(1.2).
flops(86307677).
flops_th(77479996).
nparams(1148036.0).
nparams_th(23851784).

0.99::eve.
action(reg_l2,overfitting):- eve, problem(overfitting).
action(decr_lr,inc_loss):- eve, problem(inc_loss).
action(decr_lr,high_lr):- eve, problem(high_lr).
action(inc_lr,low_lr):- eve, problem(low_lr).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.4::action(inc_neurons,underfitting):- problem(underfitting).
0.45::action(new_fc_layer):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.45::action(new_conv_layer):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
0.4::action(dec_neurons,latency):- problem(latency).
0.6::action(dec_layers,latency):- problem(latency).
0.4::action(dec_neurons,model_size):- problem(model_size).
0.5::action(dec_layers,model_size):- problem(model_size).

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
low_acc :- va(A), lacc(Tha), last(A,LTA),
                Res is LTA - 1.0, abs2(Res,Res1), Res1 > Tha.
high_loss :- vl(L), hloss(Thl), last(L,LTL), \+isclose(LTL,0,Thl).
growing_loss_trend :- l(L),add_to_UpList(L,Usl), length(L,Length_u), G is (Usl*100)/Length_u, G > 50.
up_down_acc :- a(A),add_to_UpList(A,Usa), add_to_DownList(A,Dsa), isclose(Usa,Dsa,150), Usa > 0, Dsa > 0.
up_down_loss :- l(L),add_to_UpList(L,Usl), add_to_DownList(L,Dsl), isclose(Usl,Dsl,150), Usl > 0, Dsl > 0.
to_low_lr :- area_sub(As), threshold_up(Th), As < Th.
to_high_lr :- area_sub(As), threshold_down(Th), As > Th.

% utils for hardware constraints
high_flops :- flops(V), flops_th(Th), V > Th.
high_numb_params :- nparams(V), nparams_th(Th), V > Th.


% POSSIBLE PROBLEMS
problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.
problem(low_lr) :- to_low_lr.
problem(high_lr) :- to_high_lr.

% rules for hardware constraints
problem(latency) :- high_flops.
problem(model_size) :- high_numb_params.

% QUERY ----------------------------------------------------------------------------------------------------------------
query(action(_,_)).