import numpy as np

#problem set 2 question 3


#A) -------------------
#data structure to represent decision rule, python dict (implemented as hash table), key is observation value is decision
    #ex: decisionRule = {'red': 'red', 'blue': 'red'}

#B) -----------------------------
def analzyeUrns(r1, b1, r2, b2):
    pr1 = r1/(r1+b1)
    pb1 = b1/(r1+b1)
    priors = {1:pr1, 2:pb1}
    #likelihoods
    total2 = r2 + b2 + 1
    lik = {
        # given r1
        1: {1: (r2 + 1) / total2, 2: b2 / total2},
        # given b2
        2: {1: r2 / total2, 2: (b2 + 1) / total2}
    }
    pr2 = lik[1][1]*pr1 + lik[2][1]*pb1
    pb2 = lik[1][2]*pr1 + lik[2][2] *pb1
    pObserved={1:pr2, 2:pb2}
    #Posteriors
    posterior = {
        1: {1: (lik[1][1] * pr1) / pr2, 2: (lik[2][1] * pb1) / pr2},
        2: {1: (lik[1][2] * pr1) / pb2, 2: (lik[2][2] * pb1) / pb2}
    }
    #ML Decision rule
    MLDecision = [
        1 if lik[1][1] >= lik[2][1] else 2,
        1 if lik[1][2] >= lik[2][2] else 2 
    ]
    MAPDecision = [
        1 if posterior[1][1] >= posterior[1][2] else 2,
        1 if posterior[2][1] >= posterior[2][2] else 2
    ]
    errorML = 1.0 - (posterior[1][MLDecision[0]] * pObserved[1] + posterior[2][MLDecision[1]] * pObserved[2])
    errorMAP = 1.0 - (posterior[1][MAPDecision[0]] * pObserved[1] + posterior[2][MAPDecision[1]] * pObserved[2])
    return priors, lik, posterior, MLDecision, MAPDecision, errorML, errorMAP

#partC) ---------------------------------
#The goal is to be time efficient, using NumPy array vectorization. all random draws specified by N are generated and evaluated at the same time avoiding a for loop
def runNTimes(r1, b1, r2, b2, N=100000):
    #draw N times from urn 1
    pr1 = r1/ (r1+b1)
    urn1Draws = (np.random.rand(N) >= pr1).astype(int) + 1
    #draw from urn 2
    t2 = r2+b2+1
    pr2Givendraw1 = np.where(urn1Draws==1,(r2+1)/t2, r2/t2)
    urn2Draws = (np.random.rand(N) >= pr2Givendraw1).astype(int) + 1
    _, _, _, MLRule, MAPRule, errorMLTheory, errorMAPtheory = analzyeUrns(r1,b1,r2,b2)

    MLGuess = np.where(urn2Draws == 1, MLRule[0], MLRule[1])
    MAPGuess= np.where(urn2Draws == 1, MAPRule[0], MAPRule[1])
    MLErrorRate = np.mean(MLGuess != urn1Draws)
    MAPErrorRate = np.mean(MAPGuess != urn1Draws)
    return MLErrorRate, MAPErrorRate, errorMLTheory, errorMAPtheory

def printTable(Case, results):
    print(f"Case Number: {Case}\n\tMLErrorRate: {results[0]}\tMAPErrorRate: {results[1]}\terrorMLTheory: {results[2]}\terrorMAPTheory: {results[3]}\n")
    return

if __name__ == '__main__':
    Case1 = [7,3,8,2]
    Case2 = [4,6,7,3]
    Case3 = [7,3,3,7]
    Case4 = [4,6,4,6]
    printTable('1', runNTimes(*Case1))
    printTable('2', runNTimes(*Case2))
    printTable('3', runNTimes(*Case3))
    printTable('4', runNTimes(*Case4))
    """
    Resulting Table
Case Number: 1
        MLErrorRate: 0.34466    MAPErrorRate: 0.30033   errorMLTheory: 0.34545454545454546      errorMAPTheory: 0.30000000000000004

Case Number: 2
        MLErrorRate: 0.48899    MAPErrorRate: 0.39976   errorMLTheory: 0.49090909090909085      errorMAPTheory: 0.4

Case Number: 3
        MLErrorRate: 0.52535    MAPErrorRate: 0.30289   errorMLTheory: 0.5272727272727273       errorMAPTheory: 0.30000000000000004

Case Number: 4
        MLErrorRate: 0.43435    MAPErrorRate: 0.40084   errorMLTheory: 0.4363636363636364       errorMAPTheory: 0.4
    
Comments on each Case
Case 1:
    Theoretical and calculated values align. MAP performs better than ML becasuse MAP incorporates prior probabilities of selected red or blue from URN 1 while ML only ocnsiders the likelihood of observation from Urn 2
Case 2:
    Theoretical values and calculated values also align. MAP provides more improvement because the prior probabilities are unequal. Urn 1 contains 4 red and 6 blue. MAP uses this information while ML does not making MAP more effective
Case 3:
    ML had an error rate significantly higher than MAP both theoretically and experimentally. In this case Urn 1 favors red while Urn 2 favors blue. The clor observed from Urn 2 is not a strong indication of the color transferred from Urn 1. Relying on likelihood produces ncorrect ML decisions. MAP accounts for the prior probability was red making it have a lower error rate
Case 4:
    Map still performs better similarly to case 3 but the gap is less severe

Overall:
    MAP is better than ML in these cases because it uses both the likelihood of the observed URN 2 color and prior probabilities of Urn 1 colors. MAP is expected to perform significantly better than ML in cases where the prior probabilities have significant effect on which original color is most probable.
    """
