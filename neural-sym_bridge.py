from problog.program import PrologString
from problog import get_evaluatable


class NeuralSymbolicBridge:

    def build_symbolic_model(self, facts, mode):
        """
        build logic program
        :param facts: facts to code dynamically into the symbolic program
        :param mode: diagnosis or tuning mode
        :return: compiled logic program
        """

        # create facts string for complete the symbolic model
        sym_facts = ""
        for i in facts:
            sym_facts = sym_facts + i + ".\n"

        # reading model from file
        if mode == 'diagnosis':
            f = open("symbolic/symbolic_diagnosis.pl", "r")
        else:
            f = open("symbolic/symbolic_tuning.pl", "r")
        sym_model = f.read()
        f.close()

        # compiling the assembled model
        compiled_model = get_evaluatable().create_from(PrologString(sym_facts + sym_model))

        return compiled_model

    def symbolic_reasoning(self, facts, mode):
        """
        Start symbolic reasoning
        :param facts: facts to code into the symbolic program
        :param mode: diagnosis or tuning
        :return: result of symbolic reasoning in form of dictionary
        """
        symbolic_model = self.build_symbolic_model(facts, mode)
        result = symbolic_model.evaluate()
        return result

        # use max(result, key=result.get) for return best result
