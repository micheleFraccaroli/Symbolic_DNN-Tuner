l([2.3041176795959473, 2.3036372661590576, 2.303244113922119, 2.3028833866119385, 2.3026235103607178, 2.3022243976593018, 2.302136182785034, 2.3017430305480957, 2.301281213760376, 2.300869941711426]).
sl([2.3041176795959473, 2.3039255142211914, 2.3036529541015627, 2.303345127105713, 2.3030564804077147, 2.3027236473083494, 2.3024886614990234, 2.3021904091186522, 2.3018267309753417, 2.3014440152697753]).
a([0.096220001578331, 0.09681200087070466, 0.0982072001695633, 0.09952431893348695, 0.10111459181308746, 0.10261275521278382, 0.10369565279102326, 0.10606539065132142, 0.10908723362976075, 0.11182833903249359]).
sa([0.096220001578331, 0.09769999980926514, 0.10029999911785126, 0.1014999970793724, 0.10350000113248825, 0.10486000031232834, 0.10531999915838242, 0.10961999744176865, 0.11361999809741974, 0.11593999713659286]).
vl([2.3035826683044434, 2.303222417831421, 2.302865743637085, 2.302511215209961, 2.3021538257598877, 2.3018033504486084, 2.301452159881592, 2.3010971546173096, 2.3007473945617676, 2.300400972366333]).
va([0.09669999778270721, 0.09650000184774399, 0.09539999812841415, 0.09860000014305115, 0.1054999977350235, 0.1137000024318695, 0.12309999763965607, 0.12950000166893005, 0.13220000267028809, 0.13539999723434448]).
int_loss(20.71784508228302).
int_slope(20.717926383018494).
lacc(0.15).
hloss(1.2).
flops(73788948).
flops_th(77479996).
nparams(2158525.0).
nparams_th(23851784).

0.99::eve.
action(reg_l2,overfitting):- eve, problem(overfitting).
action(decr_lr,inc_loss):- eve, problem(inc_loss).
action(decr_lr,high_lr):- eve, problem(high_lr).
action(inc_lr,low_lr):- eve, problem(low_lr).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
1.0::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
1.0::action(inc_neurons,underfitting):- problem(underfitting).
0.45::action(new_fc_layer):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.45::action(new_conv_layer):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
0.4::action(dec_neurons,latency):- problem(latency).
1.0::action(dec_layers,latency):- problem(latency).
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