l([16917.21001828995, 2.307299522781372, 2.3080129417419433, 2.3095534587860107, 2.308362205505371, 2.308367960739136, 2.3080901168823242, 2.3087843170166016, 2.3085176235198976, 2.308195806121826]).
sl([16917.21001828995, 10151.248930783082, 6091.672563646545, 3655.9273595714412, 2194.479760625067, 1317.6112035593355, 791.4899581823541, 475.8174886362191, 286.4139002311394, 172.77161846113236]).
a([0.09640000015497208, 0.10043999999761581, 0.1001839992403984, 0.09903039973974227, 0.09609824086427687, 0.09829894402742385, 0.09977936565351486, 0.09986761998815535, 0.10052057082464218, 0.10035234376794051]).
sa([0.0964, 0.1065, 0.0998, 0.0973, 0.0917, 0.1016, 0.102, 0.1, 0.1015, 0.1001]).
vl([2.309130070368449, 2.306226598739624, 2.312058578491211, 2.305710821151733, 2.309067289352417, 2.3092505518595376, 2.31324191347758, 2.305971170425415, 2.3111598904927573, 2.3151519686381024]).
va([0.097833335, 0.097833335, 0.096666664, 0.10016666, 0.096666664, 0.10016666, 0.09933333, 0.104, 0.1005, 0.1005]).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.0::action(inc_neurons,underfitting):- problem(underfitting).
0.6::action(new_fc_layer,underfitting):- problem(underfitting).
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