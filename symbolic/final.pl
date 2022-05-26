l([3.2233293056488037, 2.964824914932251, 2.7910666465759277, 2.683504581451416, 2.583247184753418, 2.5228023529052734, 2.463770866394043, 2.4322643280029297, 2.3887410163879395, 2.347156524658203]).
sl([3.2233293056488037, 3.1199275493621825, 2.9883831882476803, 2.866431745529175, 2.753157921218872, 2.6610156938934324, 2.5821177628936764, 2.522176388937378, 2.4688022399176024, 2.4201439538138425]).
a([0.12005999684333801, 0.1351480007171631, 0.15690479993820192, 0.17691887807846068, 0.19697532887458802, 0.21519319912910462, 0.23157991588783264, 0.24404394400520324, 0.2545303688788605, 0.26484621681834414]).
sa([0.12005999684333801, 0.1577800065279007, 0.18953999876976013, 0.20693999528884888, 0.227060005068779, 0.24252000451087952, 0.2561599910259247, 0.26273998618125916, 0.2702600061893463, 0.2803199887275696]).
vl([2.4501585960388184, 2.336486577987671, 2.095773220062256, 2.082829475402832, 2.0760912895202637, 2.0354421138763428, 1.9980018138885498, 1.9664812088012695, 1.930700421333313, 1.904805064201355]).
va([0.10000000149011612, 0.16580000519752502, 0.25060001015663147, 0.28519999980926514, 0.3052999973297119, 0.32179999351501465, 0.33480000495910645, 0.3440999984741211, 0.3547999858856201, 0.36399999260902405]).
int_loss(18.699287950992584).
int_slope(19.59733647108078).
lacc(0.15).
hloss(1.2).
flops(71390966).
flops_th(77479996).
nparams(1171309).
nparams_th(23851784).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
action(decr_lr,high_lr) :- eve, problem(high_lr).
action(inc_lr,low_lr) :- eve, problem(low_lr).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.0::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.0::action(inc_neurons,underfitting):- problem(underfitting).
0.45::action(new_fc_layer,underfitting,n_latency,n_model_size):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.45::action(new_conv_layer,underfitting,n_latency,n_model_size):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
0.4::action(dec_neurons,latency):- problem(latency).
0.5::action(dec_layers,latency):- problem(latency).
0.4::action(dec_neurons,model_size):- problem(model_size).
0.5::action(dec_layers,model_size):- problem(model_size).
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