l([90.29558466696739, 2.307350196123123, 2.3034198644161226, 2.304303265333176, 2.3023630623817444, 2.3033195571899414, 2.3026979694366454, 2.3027551934719086, 2.3027824227809908, 2.3027722816467286]).
sl([90.29558466696739, 55.10029087862968, 33.981542472944255, 21.310646789899824, 13.707333298892593, 9.145727802211532, 6.408515869101578, 4.76621159884971, 3.780839928422222, 3.1896128697120245]).
a([0.09399999678134918, 0.09479999840259552, 0.09728000044822693, 0.0991679995059967, 0.09910079948902129, 0.09906047947883605, 0.10423628907012938, 0.10574177456264496, 0.10824506612041473, 0.1097470410550766]).
sa([0.094, 0.096, 0.101, 0.102, 0.099, 0.099, 0.112, 0.108, 0.112, 0.112]).
vl([2.3099411797523497, 2.3061829260985056, 2.3064531469345093, 2.3038545151551566, 2.304089426199595, 2.3072438287734984, 2.3047023038069407, 2.30565305352211, 2.306713600556056, 2.3061879050731657]).
va([0.11833333, 0.11666667, 0.07666667, 0.11666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667]).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
1.0::action(inc_neurons,underfitting):- problem(underfitting).
0.0::action(new_fc_layer,underfitting):- problem(underfitting).
0.0::action(inc_batch_size,floating_loss):- problem(floating_loss).
1.0::action(decr_lr,floating_loss):- problem(floating_loss).

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

% POSSIBLE PROBLEMS
problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.

% QUERY ----------------------------------------------------------------------------------------------------------------
query(action(_,_)).