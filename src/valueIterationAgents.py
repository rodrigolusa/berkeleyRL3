# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # inpired by computeQValueFromValues and computeActionFromValues (written prior) below
        # apply computeQValues and within there replace call to computeQValues with the body of that function
        while self.iterations > 0:
            temps = self.values.copy()#save original values in temporary 
            states = self.mdp.getStates() #get all states
            for aState in states: #for each state use mdp to get all possible actions
                allActions = self.mdp.getPossibleActions(aState)
                possibleVals = []
                for action in allActions:#computeActionsFromValues do compute q-values from values
                    endStates = self.mdp.getTransitionStatesAndProbs(aState, action)
                    weighted = 0
                    for s in endStates: # for each end state calculate weigted average/q value
                        nextState = s[0] #get next state p 
                        prob = s[1] #get probability
                        reward = self.mdp.getReward(aState, action, nextState)
                        weighted += (prob * (reward + (self.discount * temps[nextState]))) 
                    possibleVals.append(weighted)
                if len(possibleVals) != 0:
                    self.values[aState] = max(possibleVals)
            self.iterations -= 1 #decrement until eventually iterations <= 0
         


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        endStates = self.mdp.getTransitionStatesAndProbs(state, action)
        weighted = 0
        for s in endStates:
            nextState = s[0]
            prob = s[1]
            reward = self.mdp.getReward(state, action, nextState)
            weighted += (prob* (reward + (self.discount * self.values[nextState])))

        return weighted

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state): #make sure game isn't over
            return None
        allActions = self.mdp.getPossibleActions(state)#get all actions
        endAction = ""
        maxSum = float("-inf") #placeholder val
        for action in allActions:
            weighted = self.computeQValueFromValues(state, action)#get wieghted average
            if (maxSum == float("-inf") and action == "") or weighted >= maxSum: #not yet assigned or bigger than current max
                endAction = action
                maxSum = weighted

        return endAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
