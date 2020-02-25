l([7.238630907058716, 2.307602174758911, 2.3015125160217287, 2.3012280340194704, 2.302318823814392, 2.303214680671692, 2.302032283782959, 2.3018310031890867, 2.301197512626648, 2.3010961923599242]).
sl([7.238630907058716, 5.266219414138794, 4.080336654891967, 3.3686932065429684, 2.942143453451538, 2.6865719443395997, 2.5327560801169433, 2.4403860493458005, 2.3847106346581395, 2.3512648577388533]).
a([0.08799999952316284, 0.09120000004768372, 0.09711999952793121, 0.09867200112342835, 0.10320320043563842, 0.1055219192123413, 0.10491315238571167, 0.10494789012012481, 0.102168733046875, 0.10290124068643189]).
sa([0.088, 0.096, 0.106, 0.101, 0.11, 0.109, 0.104, 0.105, 0.098, 0.104]).
vl([2.300443058013916, 2.3017084264755248, 2.303597569465637, 2.30685085773468, 2.3066757488250733, 2.3045328664779663, 2.3039537715911864, 2.304926300048828, 2.3051315450668337, 2.3048788833618166]).
va([0.11666667, 0.095, 0.07666667, 0.07666667, 0.07666667, 0.11833333, 0.07666667, 0.07666667, 0.07666667, 0.07666667]).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.7::action(inc_neurons,underfitting):- problem(underfitting).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).

% DIAGNOSIS SECTION ----------------------------------------------------------------------------------------------------
:- use_module(library(lists)).

% UTILITY
abs2(X,Y) :- Y is abs(X).
isclose(X,Y,W) :- D is X - Y, abs2(D,D1), D1 =< W.
add_to_UpList([_],0).
add_to_UpList([H|[H1|T]], U) :- add_to_UpList([H1|T], U1),
    (   H =< H1 ->   U is U1+1
    ;   U is U1+0
    ).


add_to_DownList([_],0).
add_to_DownList([H|[H1|T]], U) :- add_to_DownList([H1|T], U1),
    (   H > H1 -> U is U1+1
    ;   U is U1+0
    ).

% ANALYSIS
gap_tr_te_acc :- a(A), va(VA), last(A,LTA), last(VA,ScoreA),
                Res is LTA - ScoreA, abs2(Res,Res1), Res1 > 0.2.
gap_tr_te_loss :- l(L), vl(VL), last(L,LTL), last(VL,ScoreL),
                Res is LTL - ScoreL, abs2(Res,Res1), Res1 > 0.2.
low_acc :- a(A), last(A,LTA),
                Res is LTA - 1.0, abs2(Res,Res1), Res1 > 0.37.
high_loss :- l(L), last(L,LTL), isclose(LTL,0,1.7).
growing_loss_trend :- add_to_UpList(sl,Usl), length(sl,Length_u), G is (Usl*100)/Length_u, G > 50.
up_down_acc :- add_to_UpList(sa,Usa), add_to_DownList(sa,Dsa), isclose(Usa,Dsa,150), Usa > 0, Dsa > 0.
up_down_loss :- add_to_UpList(sl,Usl), add_to_DownList(sl,Dsl), isclose(Usl,Dsl,150), Usl > 0, Dsl > 0.

% POSSIBLE PROBLEMS
problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.

% QUERY ----------------------------------------------------------------------------------------------------------------
query(action(_,_)).