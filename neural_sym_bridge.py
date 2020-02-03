import sys
from problog.program import PrologString
from problog import get_evaluatable
from problog.tasks import sample


class NeuralSymbolicBridge:
    def __init__(self):
        self.all_Dfacts = ['gap_tr_te_acc', 'gap_tr_te_loss', 'low_acc', 'high_loss', 'growing_loss_trend',
                           'up_down_loss']

    def build_symbolic_model(self, facts):
        """
        build logic program
        :param facts: facts to code dynamically into the symbolic program
        :return: logic program
        """
        # reading model from file
        f = open("symbolic/symbolic_reasoning.pl", "r")
        sym_model = f.read()
        f.close()

        # create facts string for complete the symbolic model
        sym_facts = ""
        for i in self.all_Dfacts:
            if i in facts:
                sym_facts = sym_facts + i + ".\n"
            else:
                sym_facts = sym_facts + i + " :- false.\n"

        # compiling the assembled model
        # compiled_model = get_evaluatable().create_from(PrologString(sym_facts + sym_model))

        # return the assembled model
        return PrologString(sym_facts + sym_model)

    def symbolic_reasoning(self, facts):
        """
        Start symbolic reasoning
        :param facts: facts to code into the symbolic program
        :return: result of symbolic reasoning in form of list
        """
        tuning = []
        symbolic_model = self.build_symbolic_model(facts)
        result = sample.sample(symbolic_model, format='dict')
        for i in result.__next__():
            tuning.append(str(i)[str(i).find("(") + 1:str(i).find(",")])

        return tuning


if __name__ == '__main__':
    facts = ['gap_tr_te_acc', 'gap_tr_te_loss', 'low_acc']
    nsb = NeuralSymbolicBridge()
    res_tuning = nsb.symbolic_reasoning(facts)
    print(res_tuning)
