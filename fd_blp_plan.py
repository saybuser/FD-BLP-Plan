from krrt.utils import get_opts

import math

import cplex
from cplex.exceptions import CplexError

def readBNN(directory):
    
    layers = []
    weights = {}
    BNNFile = open(directory,"r")
    data = BNNFile.read().splitlines()
    
    layerSizeIndex = 1
    input = 0
    output = 0
    layer = 0
    out = 0
    
    for index, dat in enumerate(data):
        if (index == layerSizeIndex):
            input, output = dat.split(",")
            layerSizeIndex = int(output) + layerSizeIndex + 1
            layers.append([int(input),int(output)])
            layer += 1
            out = 0
        else:
            for inp in range(int(input)):
                weight = 0
                if (dat[int(inp)] == '0'):
                    weight = -1
                elif (dat[int(inp)] == '1'):
                    weight = 1
                else:
                    weight = 0
                weights[(layer-1, inp, out)] = weight
            out += 1

    return weights, layers

def readNormalization(directory, layers):
    
    nLayers = len(layers)
    normalization = []
    NormalizationFile = open(directory,"r")
    data = NormalizationFile.read().splitlines()
    
    for index, dat in enumerate(data):
        if index > 0:
            normalization.append(dat.split(","))

    return normalization

def readInitial(directory):
    
    initial = []
    initialFile = open(directory,"r")
    data = initialFile.read().splitlines()
    
    for dat in data:
        initial.append(dat.split(","))

    return initial

def readGoals(directory):
    
    goals = []
    goalsFile = open(directory,"r")
    data = goalsFile.read().splitlines()
    
    for dat in data:
        goals.append(dat.split(","))

    return goals

def readConstraints(directory):
    
    constraints = []
    constraintsFile = open(directory,"r")
    data = constraintsFile.read().splitlines()
    
    for dat in data:
        constraints.append(dat.split(","))

    return constraints

def readTransitions(directory):
    
    transitions = []
    transitionsFile = open(directory,"r")
    data = transitionsFile.read().splitlines()
    
    for dat in data:
        transitions.append(dat.split(","))
    
    return transitions

def readReward(directory):
    
    reward = []
    rewardFile = open(directory,"r")
    data = rewardFile.read()
    
    reward = data.split(",")
    
    return reward

def readVariables(directory):
    
    A = []
    AData = []
    S = []
    SData = []
    SLabel = []
    
    variablesFile = open(directory,"r")
    data = variablesFile.read().splitlines()
    
    for dat in data:
        variables = dat.split(",")
        for var in variables:
            if "action:" in var or "action_data:" in var:
                if "_data:" in var:
                    AData.append(var.replace("action_data: ",""))
                    A.append(var.replace("action_data: ",""))
                else:
                    A.append(var.replace("action: ",""))
            else:
                if "_data:" in var or "_data_label:" in var:
                    if "_label:" in var:
                        SData.append(var.replace("state_data_label: ",""))
                        SLabel.append(var.replace("state_data_label: ",""))
                        S.append(var.replace("state_data_label: ",""))
                    else:
                        SData.append(var.replace("state_data: ",""))
                        S.append(var.replace("state_data: ",""))
                else:
                    if "_label:" in var:
                        SLabel.append(var.replace("state_label: ",""))
                        S.append(var.replace("state_label: ",""))
                    else:
                        S.append(var.replace("state: ",""))

    return A, AData, S, SData, SLabel

def encode_fd_blp_plan(domain, instance, horizon, optimize):
    
    weights, layers = readBNN("./bnn/bnn_"+domain+"_"+instance+".txt")
    
    normalization = readNormalization("./normalization/normalization_"+domain+"_"+instance+".txt", layers)
    
    initial = readInitial("./translation/initial_"+domain+"_"+instance+".txt")
    goals = readGoals("./translation/goals_"+domain+"_"+instance+".txt")
    constraints = readConstraints("./translation/constraints_"+domain+"_"+instance+".txt")
    A, AData, S, SData, SLabel = readVariables("./translation/pvariables_"+domain+"_"+instance+".txt")
    
    nHiddenLayers = len(layers)-1
    VARINDEX = 0
    
    #SLabel = S[:layers[len(layers)-1][1]] #SLabel = S Sometimes, you can also assume this is true.
    
    transitions = []
    if len(SLabel) < len(S):
        transitions = readTransitions("./translation/transitions_"+domain+"_"+instance+".txt")
    reward = []
    if optimize == "True":
        reward = readReward("./translation/reward_"+domain+"_"+instance+".txt")
    
    # CPLEX
    c = cplex.Cplex()
    vartypes = ""
    colnames = []
    objcoefs = []

    # Create vars for each action a, time step t
    x = {}
    for a in A:
        for t in range(horizon):
            x[(a,t)] = VARINDEX
            colnames.append(str(x[(a,t)]))
            objcoefs.append(0.0)
            vartypes += "B"
            VARINDEX += 1

    # Create vars for each state a, time step t
    y = {}
    for s in S:
        for t in range(horizon+1):
            y[(s,t)] = VARINDEX
            colnames.append(str(y[(s,t)]))
            objcoefs.append(0.0)
            vartypes += "B"
            VARINDEX += 1

    # Create vars for each activation node z at depth d, width w, time step t
    z = {}
    for t in range(horizon):
        for d in range(nHiddenLayers):
            for w in range(layers[d][1]):
                z[(d,w,t)] = VARINDEX
                colnames.append(str(z[(d,w,t)]))
                objcoefs.append(0.0)
                vartypes += "B"
                VARINDEX += 1

    if optimize == "True":
        for t in range(horizon):
            for var in reward:
                if var in A or var[1:] in A:
                    if var[0] == "~":
                        objcoefs[colnames.index(str(x[(var[1:],t)]))] = 1.0
                    else:
                        objcoefs[colnames.index(str(x[(var,t)]))] = -1.0
                else:
                    if var[0] == "~":
                        objcoefs[colnames.index(str(y[(var[1:],t+1)]))] = 1.0
                    else:
                        objcoefs[colnames.index(str(y[(var,t+1)]))] = -1.0
        c.variables.add(obj=objcoefs, types=vartypes, names=colnames)
    else:
        c.variables.add(types=vartypes, names=colnames)

    # Constraints
    for t in range(horizon+1):
        for constraint in constraints:
            variables = constraint[:-2]
            literals = []
            coefs = []
            RHS = 0
            if set(A).isdisjoint(variables) or t < horizon: # for the last time step, only consider constraints that include states variables-only
                for var in variables:
                    if var in A or var[1:] in A:
                        if var[0] == "~":
                            literals.append(x[(var[1:],t)])
                            coefs.append(-1.0)
                            RHS -= 1
                        else:
                            literals.append(x[(var,t)])
                            coefs.append(1.0)
                    else:
                        if var[0] == "~":
                            literals.append(y[(var[1:],t)])
                            coefs.append(-1.0)
                            RHS -= 1
                        else:
                            literals.append(y[(var,t)])
                            coefs.append(1.0)
                RHS += int(constraint[len(constraint)-1])
                if "<=" == constraint[len(constraint)-2]:
                    row = [ [ literals, coefs ] ]
                    c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
                elif ">=" == constraint[len(constraint)-2]:
                    row = [ [ literals, coefs ] ]
                    c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
                else:
                    row = [ [ literals, coefs ] ]
                    c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])

    # Known Transitions
    for t in range(horizon):
        for transition in transitions:
            variables = transition[:-2]
            literals = []
            coefs = []
            RHS = 0
            for var in variables:
                if var in A or var[1:] in A:
                    if var[0] == "~":
                        literals.append(x[(var[1:],t)])
                        coefs.append(-1.0)
                        RHS -= 1
                    else:
                        literals.append(x[(var,t)])
                        coefs.append(1.0)
                else:
                    if var[0] == "~":
                        if var[len(var)-1] == "'":
                            literals.append(y[(var[1:-1],t+1)])
                            coefs.append(-1.0)
                            RHS -= 1
                        else:
                            literals.append(y[(var[1:],t)])
                            coefs.append(-1.0)
                            RHS -= 1
                    else:
                        if var[len(var)-1] == "'":
                            literals.append(y[(var[:-1],t+1)])
                            coefs.append(1.0)
                        else:
                            literals.append(y[(var,t)])
                            coefs.append(1.0)
            RHS += int(transition[len(transition)-1])
            if "<=" == transition[len(transition)-2]:
                row = [ [ literals, coefs ] ]
                c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
            elif ">=" == transition[len(transition)-2]:
                row = [ [ literals, coefs ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
            else:
                row = [ [ literals, coefs ] ]
                c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])

    # Set initial state
    for init in initial:
        variables = init[:-2]
        literals = []
        coefs = []
        RHS = 0
        for var in variables:
            if var[0] == "~":
                literals.append(y[(var[1:],0)])
                coefs.append(-1.0)
                RHS -= 1
            else:
                literals.append(y[(var,0)])
                coefs.append(1.0)
        RHS += int(init[len(init)-1])
        if "<=" == init[len(init)-2]:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
        elif ">=" == init[len(init)-2]:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
        else:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])

    # Set goal state
    for goal in goals:
        variables = goal[:-2]
        literals = []
        coefs = []
        RHS = 0
        for var in variables:
            if var[0] == "~":
                literals.append(y[(var[1:],horizon)])
                coefs.append(-1.0)
                RHS -= 1
            else:
                literals.append(y[(var,horizon)])
                coefs.append(1.0)
        RHS += int(goal[len(goal)-1])
        if "<=" == goal[len(goal)-2]:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
        elif ">=" == goal[len(goal)-2]:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
        else:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])

    # Set node activations
    for t in range(horizon):
        for d in range(nHiddenLayers):
            for out in range(layers[d][1]):
                inputLiterals = []
                coefs = []
                RHS = 0
                layersize = 0
                
                if d == 0: # input is state or actions
                    for inp, a in enumerate(AData):
                        if weights[(d, inp, out)] > 0:
                            inputLiterals.append(x[(a,t)])
                            coefs.append(1.0)
                            layersize += 1
                        elif weights[(d, inp, out)] < 0:
                            inputLiterals.append(x[(a,t)])
                            coefs.append(-1.0)
                            RHS -= 1
                            layersize += 1
                    for i, s in enumerate(SData):
                        inp = i + len(AData)
                        if weights[(d, inp, out)] > 0:
                            inputLiterals.append(y[(s,t)])
                            coefs.append(1.0)
                            layersize += 1
                        elif weights[(d, inp, out)] < 0:
                            inputLiterals.append(y[(s,t)])
                            coefs.append(-1.0)
                            RHS -= 1
                            layersize += 1
                else:
                    for inp in range(layers[d][0]):
                        if weights[(d, inp, out)] > 0:
                            inputLiterals.append(z[(d-1,inp,t)])
                            coefs.append(1.0)
                            layersize += 1
                        elif weights[(d, inp, out)] < 0:
                            inputLiterals.append(z[(d-1,inp,t)])
                            coefs.append(-1.0)
                            RHS -= 1
                            layersize += 1
            
                positive_threshold = int(math.ceil(layersize/2.0 + float(normalization[d][out])/2.0))
                negative_threshold = layersize - positive_threshold + 1
                
                if positive_threshold >= layersize + 1:
                    row = [ [[z[(d,out,t)]], [1.0]] ]
                    c.linear_constraints.add(lin_expr=row, senses="E", rhs=[0.0])
                elif negative_threshold >= layersize + 1:
                    row = [ [[z[(d,out,t)]], [1.0]] ]
                    c.linear_constraints.add(lin_expr=row, senses="E", rhs=[1.0])
                else:
                    inputLiterals.append(z[(d,out,t)])

                    coefs.append(-1.0*positive_threshold)
                    row = [ [ inputLiterals, coefs ] ]
                    c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])

                    RHS = -1.0*layersize - RHS + negative_threshold
                    coefs = [-i for i in coefs]
                    coefs[len(coefs)-1] = negative_threshold
                    row = [ [ inputLiterals, coefs ] ]
                    c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])

    # Predict the next state using BNNs
    for t in range(horizon):
        d = nHiddenLayers
        for out, s in enumerate(SLabel):
            inputLiterals = []
            coefs = []
            RHS = 0
            layersize = 0
            
            for inp in range(layers[d][0]):
                if weights[(d, inp, out)] > 0:
                    inputLiterals.append(z[(d-1,inp,t)])
                    coefs.append(1.0)
                    layersize += 1
                elif weights[(d, inp, out)] < 0:
                    inputLiterals.append(z[(d-1,inp,t)])
                    coefs.append(-1.0)
                    RHS -= 1
                    layersize += 1
        
            positive_threshold = int(math.ceil(layersize/2.0 + float(normalization[d][out])/2.0))
            negative_threshold = layersize - positive_threshold + 1
            
            if positive_threshold >= layersize + 1:
                row = [ [[y[(s,t+1)]], [1.0]] ]
                c.linear_constraints.add(lin_expr=row, senses="E", rhs=[0.0])
            elif negative_threshold >= layersize + 1:
                row = [ [[y[(s,t+1)]], [1.0]] ]
                c.linear_constraints.add(lin_expr=row, senses="E", rhs=[1.0])
            else:
                inputLiterals.append(y[(s,t+1)])
                
                coefs.append(-1.0*positive_threshold)
                row = [ [ inputLiterals, coefs ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
                
                RHS = -1.0*layersize - RHS + negative_threshold
                coefs = [-i for i in coefs]
                coefs[len(coefs)-1] = negative_threshold
                row = [ [ inputLiterals, coefs ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])

    c.parameters.threads.set(1)
    
    c.solve()

    solX = c.solution.get_values()
    
    for t in range(horizon):
        for a in A:
            if(solX[x[(a,t)]] + 0.000001 >= 1.0):
                print("%s at time: %d" % (a,t))

    return

if __name__ == '__main__':
    #import os
    myargs, flags = get_opts()
    
    setDomain = False
    setInstance = False
    setHorizon = False
    setObjective = False
    for arg in myargs:
        if arg == "-d":
            domain = myargs[(arg)]
            setDomain = True
        elif arg == "-i":
            instance = myargs[(arg)]
            setInstance = True
        elif arg == "-h":
            horizon = myargs[(arg)]
            setHorizon = True
        elif arg == "-o":
            optimize = myargs[(arg)]
            setObjective = True

    if setDomain and setInstance and setHorizon and setObjective:
        encode_fd_blp_plan(domain, instance, int(horizon), optimize)
    elif not setDomain:
        print 'Domain is not provided.'
    elif not setInstance:
        print 'Instance is not provided.'
    elif not setHorizon:
        print 'Horizon is not provided.'
    else:
        print 'Optimization setting is not provided.'


    #encode_fd_blp_plan("navigation", "3x3", 4, "False")
    #encode_fd_blp_plan("navigation", "4x4", 5, "False")
    #encode_fd_blp_plan("navigation", "5x5", 8, "False")

    #encode_fd_blp_plan("inventory", "1", 7, "True")
    #encode_fd_blp_plan("inventory", "2", 8, "True")

    #encode_fd_blp_plan("sysadmin", "5", 4, "False")
