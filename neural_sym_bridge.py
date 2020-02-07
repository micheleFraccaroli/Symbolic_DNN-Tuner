import sys
from problog.program import PrologString
from problog import get_evaluatable
from problog.tasks import sample


class NeuralSymbolicBridge:
    def __init__(self):
        self.initial_facts = ['l', 'sl', 'a', 'sa', 'vl', 'va']

    def build_symbolic_model(self, facts):
        """
        build logic program
        :param facts: facts to code dynamically into the symbolic program
        :return: logic program
        """
        # reading model from file
        f = open("symbolic/symbolic_analysis.pl", "r")
        sym_model = f.read()
        f.close()

        # create facts string for complete the symbolic model
        sym_facts = ""
        for fa, i in zip(facts, self.initial_facts):
            sym_facts = sym_facts + i + "(" + str(fa) + ").\n"

        # return the assembled model
        return PrologString(sym_facts + sym_model)

    def symbolic_reasoning(self, facts, diagnosis_logs, tuning_logs):
        """
        Start symbolic reasoning
        :param facts: facts to code into the symbolic program
        :return: result of symbolic reasoning in form of list
        """
        tuning = []
        diagnosis = []
        symbolic_model = self.build_symbolic_model(facts)
        result = sample.sample(symbolic_model, format='dict')
        for i in result.__next__():
            tuning.append(str(i)[str(i).find("(") + 1:str(i).find(",")])
            diagnosis.append(str(i)[str(i).find(",") + 1:str(i).find(")")])

        diagnosis_logs.write(str(diagnosis))
        tuning_logs.write(str(tuning))
        return tuning
