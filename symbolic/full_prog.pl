:- use_module(library(lists)).

% DIAGNOSIS SECTION ----------------------------------------------------------------------------------------------------

% utility
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

% analysis
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

% problems
problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.


query(action(_,_)).