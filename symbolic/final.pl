l([93893.87750485039, 2.3134915409088133, 2.3076352507273357, 2.3083150672912596, 2.307574785232544, 2.3059372844696044, 2.3058515307108562, 2.3054372520446775, 2.3065709800720215, 2.3057148609161375, 2.305467665354411, 2.304016045252482, 2.3055046825408936, 2.303811364491781, 2.30655961672465]).
sl([93893.87750485039, 56337.2518995266, 33803.27419381625, 20282.887842316668, 12170.655735304093, 7303.315816096244, 4382.911830270031, 2630.6692730628365, 1579.3241922297307, 948.5168012822048, 570.0322678354646, 342.94096711937976, 206.6867821446442, 124.93359383258324, 75.8827801462398]).
a([0.09666666388511658, 0.0978666663169861, 0.10001999855041505, 0.09967866671085357, 0.0988405324935913, 0.09977098571777343, 0.09896259169292448, 0.09854422307319641, 0.10092653510753631, 0.10005592213740538, 0.10140021983784789, 0.10164013113976927, 0.10208408001900561, 0.1028837827673009, 0.10016360389175719]).
sa([0.096666664, 0.09966667, 0.10325, 0.09916667, 0.09758333, 0.101166666, 0.09775, 0.09791667, 0.1045, 0.09875, 0.10341667, 0.102, 0.10275, 0.10408334, 0.096083336]).
vl([2.3092510255177814, 2.308010222752889, 2.3093826948801675, 2.3069260215759275, 2.308865525563558, 2.305235933303833, 2.3038418394724527, 2.304394441604614, 2.3105950921376546, 2.3039655291239423, 2.306784475962321, 2.305615660349528, 2.3061359742482503, 2.3062979062398274, 2.303536246617635]).
va([0.10016666, 0.09883333, 0.096666664, 0.096666664, 0.102, 0.09883333, 0.104, 0.09883333, 0.104, 0.09883333, 0.096666664, 0.097833335, 0.097833335, 0.09933333, 0.100833334]).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
1.0::action(inc_neurons,underfitting):- problem(underfitting).
1.0::action(inc_batch_size,floating_loss):- problem(floating_loss).
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