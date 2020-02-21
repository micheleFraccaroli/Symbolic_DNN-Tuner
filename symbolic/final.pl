l([88009.44214679957, 5.403008623123169, 2.3571670961380007, 2.3059266567230225, 2.3034933853149413, 2.3050840997695925, 2.303830931186676, 2.305596158504486, 2.3022201013565065, 2.3032325601577757]).
sl([88009.44214679957, 52807.82649152899, 31685.638761755847, 19012.3056277162, 11408.304773983846, 6845.904898030215, 4108.464471190604, 2466.000921177764, 1480.521440747201, 889.2341574723837]).
a([0.09600000083446503, 0.09200000166893005, 0.09520000159740448, 0.10112000072002411, 0.10547200181484223, 0.1080832024717331, 0.10444992126846314, 0.09586995149745942, 0.10232197228130341, 0.10099318315420533]).
sa([0.096, 0.086, 0.1, 0.11, 0.112, 0.112, 0.099, 0.083, 0.112, 0.099]).
vl([10.83560128211975, 2.3490007718404136, 2.314363165696462, 2.314691476027171, 2.3092636982599895, 2.3067763368288676, 2.3044572949409483, 2.3070836861928306, 2.307464317480723, 2.3039807796478273]).
va([0.11833333, 0.105, 0.105, 0.07666667, 0.07666667, 0.07666667, 0.11666667, 0.07666667, 0.07666667, 0.07666667]).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.111111111111111::action(inc_neurons,underfitting):- problem(underfitting).
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