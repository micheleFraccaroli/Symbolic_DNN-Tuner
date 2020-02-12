from problog.program import PrologString
from problog.logic import Term
from problog.learning import lfi


class LfiIntegration:
    def __init__(self):
        self.experience = []
        self.ove = Term('ove')
        self.und = Term('und')
        self.flo = Term('flo')
        self.reg_l2 = Term('reg_l2')
        self.inc_dropout = Term('inc_dropout')
        self.decr_lr = Term('decr_lr')
        self.inc_neurons = Term('inc_neurons')
        self.inc_batch_size = Term('inc_batch_size')
        self.data_augmentation = Term('data_augmentation')
        self.overfitting = Term('overfitting')
        self.underfitting = Term('underfitting')
        self.inc_loss = Term('inc_loss')
        self.floating_loss = Term('floating_loss')
        self.action1 = Term('action', self.reg_l2, self.overfitting)
        self.action2 = Term('action', self.inc_dropout, self.overfitting)
        self.action3 = Term('action', self.decr_lr, self.underfitting)
        self.action4 = Term('action', self.inc_neurons, self.underfitting)
        self.action5 = Term('action', self.decr_lr, self.inc_loss, self.inc_loss)
        self.action6 = Term('action', self.inc_batch_size, self.floating_loss)
        self.action7 = Term('action', self.decr_lr, self.floating_loss)
        self.action8 = Term('action', self.data_augmentation, self.overfitting)

    def create_evidence(self, t, d, bool):
        t1 = Term(str(t))
        t2 = Term(str(d))
        prob = Term(d[:3])
        action = Term('action', t1, t2)
        evidence1 = (action, bool)
        evidence2 = (prob, bool)
        return evidence1, evidence2

    def evidence(self, improve, tuning, diagnosis):
        evidence = []
        for t, d in zip(tuning, diagnosis):
            if improve:
                e1, e2 = self.create_evidence(t, d, True)
            else:
                e1, e2 = self.create_evidence(t, d, False)
            evidence.append(e1)
            #evidence.append(e2)
        return evidence

    def learning(self, improve, tuning, diagnosis):
        evidence = self.evidence(improve, tuning, diagnosis)
        self.experience.append(evidence)
        f1 = open("symbolic/lfi.pl", "r")
        to_learn = f1.read()
        f1.close()
        _, weights, _, _, lfi_problem = lfi.run_lfi(PrologString(to_learn), self.experience)

        # temp1 = to_learn.splitlines()
        # temp2 = []
        # for i, t in zip(weights, temp1):
        #     temp2.append('t(' + str(i) + ')::' + t[t.find("::") + 2:t.find("\n")] + ".")
        #
        # new_f1 = open("symbolic/lfi.pl", "w")
        # for i in temp2:
        #     new_f1.write(i + "\n")
        # new_f1.close()

        return weights, lfi_problem
