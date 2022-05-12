l([1.5693204402923584, 1.244767189025879, 1.1111133098602295, 1.017224669456482, 0.9445616602897644, 0.8787964582443237, 0.8188056945800781, 0.7631458044052124, 0.7079256772994995, 0.6553720831871033, 0.6012533903121948, 0.550969660282135, 0.5014815330505371, 0.4540807008743286, 0.40486809611320496, 0.3572823703289032, 0.314586341381073, 0.2718889117240906, 0.2344258725643158, 0.19753065705299377]).
sl([1.5693204402923584, 1.4394991397857666, 1.3081448078155518, 1.1917767524719238, 1.0928907155990601, 1.0072530126571655, 0.9318740854263305, 0.8643827730178834, 0.8017999347305298, 0.7432287941131592, 0.6864386325927735, 0.6322510436685181, 0.5799432394213258, 0.5295982240025269, 0.47970617284679806, 0.43073665183964005, 0.38427652765621323, 0.33932148128336415, 0.2973632377957448, 0.25743020549864437]).
a([0.44707998633384705, 0.4948479950428009, 0.5430927884578705, 0.5850876696109771, 0.621484597158432, 0.6514747617435455, 0.6787968681926726, 0.7021181229755401, 0.7234868791106415, 0.7445161215307159, 0.7648936672116424, 0.7848641969624225, 0.8027745066799559, 0.8206247078608178, 0.8383668176669301, 0.8555000968181146, 0.8723640648962887, 0.8884744291568893, 0.9034126547017752, 0.9171355931186115]).
sa([0.44707998633384705, 0.5665000081062317, 0.6154599785804749, 0.6480799913406372, 0.6760799884796143, 0.6964600086212158, 0.7197800278663635, 0.7371000051498413, 0.7555400133132935, 0.7760599851608276, 0.7954599857330322, 0.8148199915885925, 0.8296399712562561, 0.8474000096321106, 0.8649799823760986, 0.8812000155448914, 0.8976600170135498, 0.9126399755477905, 0.925819993019104, 0.937720000743866]).
vl([1.378305435180664, 1.1649036407470703, 1.0907777547836304, 1.0374011993408203, 1.005427360534668, 0.9403932690620422, 0.9025788903236389, 0.9043735265731812, 0.8691104650497437, 0.844574511051178, 0.8446826338768005, 0.8697680830955505, 0.8511815667152405, 0.8365204334259033, 0.8619421720504761, 0.8748436570167542, 0.8922551870346069, 0.9280951023101807, 0.9554562568664551, 1.0105091333389282]).
va([0.5120999813079834, 0.5932000279426575, 0.6234999895095825, 0.6414999961853027, 0.6542999744415283, 0.6797999739646912, 0.6904000043869019, 0.6869999766349792, 0.7056999802589417, 0.7160999774932861, 0.7218999862670898, 0.7202000021934509, 0.7253000140190125, 0.7347999811172485, 0.732200026512146, 0.7361000180244446, 0.7379000186920166, 0.7404999732971191, 0.7347000241279602, 0.7372999787330627]).
int_loss(17.868692994117737).
int_slope(22.693738400936127).
lacc(0.15).
hloss(1.2).
flops(157890876).
flops_th(77479996).
nparams(2625098.0).
nparams_th(23851784).

0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
action(decr_lr,high_lr) :- eve, problem(high_lr).
action(inc_lr,low_lr) :- eve, problem(low_lr).
0.5::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
1.0::action(inc_neurons,underfitting):- problem(underfitting).
0.45::action(new_fc_layer,underfitting):- problem(underfitting).
0.45::action(new_conv_layer,underfitting):- problem(underfitting).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
0.5::action(dec_neurons,latency):- problem(latency).
0.5::action(dec_layers,latency):- problem(latency).
0.5::action(dec_neurons,model_size):- problem(model_size).
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