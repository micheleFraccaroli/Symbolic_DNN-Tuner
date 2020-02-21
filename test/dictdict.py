import sys

res = {}
problems = []


symbolic_evaluation = {
    'action(data_augmentation,overfitting)': 0.6,
    'action(decr_lr,underfitting)': 0,
    'action(inc_dropout,overfitting)': 0.4,
    'action(inc_neurons,underfitting)': 0.7,
    'action(reg_l2,overfitting)': 0.99
}

for i in symbolic_evaluation.keys():
    problems.append(str(i)[str(i).find(",") + 1:str(i).find(")")])

problems = list(dict.fromkeys(problems))

for i in problems:
    inner = {}
    for j in symbolic_evaluation.keys():
        if i in j:
            inner[str(j)[str(j).find("(") + 1:str(j).find(",")]] = symbolic_evaluation[j]
    res[i] = inner


for i in res.keys():
    print(i)
    print(max(res[i]))
    print(res[i][max(res[i])])
    print()