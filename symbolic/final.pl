l([16.997553443908693, 2.317287826538086, 2.3069767570495605, 2.302727098464966, 2.3014306831359863, 2.3016069316864014, 2.3008280181884766, 2.3019266605377195, 2.3015078258514405, 2.3010629081726073]).
sl([16.997553443908693, 11.125447196960451, 7.598059020996095, 5.479926251983644, 4.208528024444581, 3.445759587341309, 2.987786959680176, 2.713442840023194, 2.5486688343544923, 2.4496264638817387]).
a([0.10199999809265137, 0.10519999861717225, 0.10792000055313111, 0.10955200171470643, 0.11053120241165161, 0.11111872282981873, 0.111471235080719, 0.11168274243125915, 0.11180964684158325, 0.1118857894877777]).
sa([0.102, 0.11, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]).
vl([2.3163418134053546, 2.311760965983073, 2.3061927954355874, 2.305898729960124, 2.3060513019561766, 2.307906087239583, 2.3064726988474527, 2.3075278917948405, 2.304596265157064, 2.3062970638275146]).
va([0.11666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667, 0.07666667]).

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