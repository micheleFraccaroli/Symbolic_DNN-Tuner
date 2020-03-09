l([3.4257152309417727, 3.084995059967041, 2.959659553050995, 2.81320246219635, 2.680634226322174, 2.66051975107193, 2.664046021938324, 2.546303625583649, 2.439545150279999, 2.4457686955928803, 2.461737428188324, 2.5026182246208193, 2.335919292449951, 2.435777594089508, 2.3947580318450927, 2.3431359770298004, 2.2850220355987547]).
sl([3.4257152309417727, 3.2894271625518803, 3.157520118751526, 3.0197930561294557, 2.884129524206543, 2.7946856149526975, 2.742429777746948, 2.663979316881629, 2.5742056502409767, 2.522830868381738, 2.4983934923043725, 2.500083385230951, 2.434417748118551, 2.434961686506934, 2.4188802246421974, 2.3885825255972386, 2.3471583295978453]).
a([0.11500000208616257, 0.1413999989628792, 0.16003999680280687, 0.17802399736642838, 0.20361439789533614, 0.2093686366391182, 0.21802118446302415, 0.23441271201295855, 0.2454476250143242, 0.25166857796498493, 0.2562011513566277, 0.25892069539161333, 0.2725524222894419, 0.2751314590003436, 0.2794788807407823, 0.2872873264417533, 0.28957240091952585]).
sa([0.115, 0.181, 0.188, 0.205, 0.242, 0.218, 0.231, 0.259, 0.262, 0.261, 0.263, 0.263, 0.293, 0.279, 0.286, 0.299, 0.293]).
vl([2.27810933192571, 2.2569942673047385, 2.2957216284491797, 2.4296573599179587, 2.5598066051801047, 2.803497632344564, 2.989912688732147, 3.087525169054667, 3.1626018484433494, 3.209883689880371, 3.277665456136068, 3.282602588335673, 3.2577742536862693, 3.075914003632285, 3.057855010032654, 2.8848150769869485, 2.6568374633789062]).
va([0.105, 0.135, 0.15833333, 0.12666667, 0.11833333, 0.096666664, 0.093333334, 0.108333334, 0.108333334, 0.1, 0.105, 0.11, 0.11666667, 0.12166667, 0.13, 0.15666667, 0.18833333]).
int_loss(46.099700675769284).
int_slope(128.0).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
action(decr_lr,high_lr) :- eve, problem(high_lr).
0.0::action(inc_dropout,overfitting):- problem(overfitting).
0.0::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.1::action(inc_neurons,underfitting):- problem(underfitting).
0.3::action(new_fc_layer,underfitting):- problem(underfitting).
0.6::action(new_conv_layer,underfitting):- problem(underfitting).
0.0::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.6::action(decr_lr,floating_loss):- problem(floating_loss).

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
low_acc :- a(A), last(A,LTA),
                Res is LTA - 1.0, abs2(Res,Res1), Res1 > 0.30.
high_loss :- l(L), last(L,LTL), \+isclose(LTL,0,1.7).
growing_loss_trend :- l(L),add_to_UpList(L,Usl), length(L,Length_u), G is (Usl*100)/Length_u, G > 50.
up_down_acc :- a(A),add_to_UpList(A,Usa), add_to_DownList(A,Dsa), isclose(Usa,Dsa,150), Usa > 0, Dsa > 0.
up_down_loss :- l(L),add_to_UpList(L,Usl), add_to_DownList(L,Dsl), isclose(Usl,Dsl,150), Usl > 0, Dsl > 0.
to_low_lr :- area_sub(As), threshold_up(Th), As < Th.
to_high_lr :- area_sub(As), threshold_down(Th), As > Th.

% POSSIBLE PROBLEMS
problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.
problem(low_lr) :- to_low_lr.
problem(high_lr) :- to_high_lr.

% QUERY ----------------------------------------------------------------------------------------------------------------
query(action(_,_)).