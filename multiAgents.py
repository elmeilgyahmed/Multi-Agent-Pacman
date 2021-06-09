# multiAgents.py
# --------------
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

from util import manhattanDistance
from game import Directions
import random, util
from math import inf

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        new_ghostPos=successorGameState.getGhostPositions()
        newFood = successorGameState.getFood()
        current_listFood=currentGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score=successorGameState.getScore()

        "*** YOUR CODE HERE ***"
        nearestFood = 999999 #or inf
        for food in successorGameState.getFood().asList():nearestFood = min(nearestFood, manhattanDistance(newPos, food))#get the nearest food and store it in nearestFood
        # avoid ghost if too close
        # print(nearestFood)
        # print(successorGameState.getScore())
        return successorGameState.getScore() + nearestFood**(-1) #increase the score of the succesor by 1/nearest food for the nearest food the the score is increasd

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0)[0] # 0(indexAgent) for pacman,0 intiate for depth [0] to get the action from  action_corr_Succ 

    def minimax(self, gameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == self.depth * gameState.getNumAgents(): #check if isLose,isWon,or it exceed the max depth
           return self.evaluationFunction(gameState)
        if agentIndex == 0:  # agentIndex=0 this for pacman call maxval
            return self.maxval(gameState, agentIndex, depth)[1]
        else: # else if it ghost we call minval
            return self.minval(gameState, agentIndex, depth)[1]

    def maxval(self, gameState, agentIndex, depth):
        maxi = ("maxi any thing",-inf)# intiate maxi -inf ,,,add"maxi" just to use entry[1] with action_corr_Succ to get action will get clear below,,,  to get agentIndex=(depth+1)%gameState.getNumAgents() i get this formula from internet 
        for action in gameState.getLegalActions(agentIndex):# for every action in legal actions get max of (maxi,minimax(gamestate succsor),agent index,depth+1))
            action_corr_Succ = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),agentIndex+1,depth+1))
            #print(depth)
            #print(gameState.getNumAgents())
            #print((depth + 1)%gameState.getNumAgents())
            maxi = max(maxi,action_corr_Succ,key=lambda entry:entry[1])
        return maxi
    def minval(self, gameState, agentIndex, depth):
        mini = ("mini any thing ",inf)# intiate mini inf
        for action in gameState.getLegalActions(agentIndex):
            total_agents=gameState.getNumAgents()
            action_corr_Succ = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),(depth + 1)%total_agents,depth+1))
            #print(depth)
            #print(gameState.getNumAgents())
            #print((depth + 1)%gameState.getNumAgents())
            mini = min(mini,action_corr_Succ,key=lambda entry:entry[1])
        return mini
       

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0, -inf, inf)[0]
    def alphabeta(self, gameState, agentIndex, depth,alpha,beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth * gameState.getNumAgents():
           return self.evaluationFunction(gameState)
        if agentIndex == 0: 
            return self.maxval(gameState, agentIndex, depth,alpha,beta)[1]
        else: # else if it ghost we return minval
            return self.minval(gameState, agentIndex, depth,alpha,beta)[1]

    def maxval(self, gameState, agentIndex, depth,alpha,beta):
        maxi = ("maxi any thing ",-inf) # intiate maxi -inf ,,, to get agentIndex=(depth+1)%gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):# for every action in legal actions get max of (maxi,minimax(gamestate succsor),agent index,depth+1))
            total_agents=gameState.getNumAgents()
            action_corr_Succ = (action,self.alphabeta(gameState.generateSuccessor(agentIndex,action),(depth + 1)%total_agents,depth+1,alpha,beta))
            maxi = max(maxi,action_corr_Succ,key=lambda entry:entry[1])# get only action_corr_succ[1] and maxi[1]

          
            if maxi[1] > beta: return maxi
            else: alpha = max(alpha,maxi[1])

        return maxi
    def minval(self, gameState, agentIndex, depth,alpha,beta):
        mini = ("mini any thing",inf)# intiate mini inf
        for action in gameState.getLegalActions(agentIndex):
            total_agents=gameState.getNumAgents()
            action_corr_Succ = (action,self.alphabeta(gameState.generateSuccessor(agentIndex,action),(depth + 1)%total_agents,depth+1,alpha,beta))
            mini = min(mini,action_corr_Succ,key=lambda entry:entry[1])

          
            if mini[1] < alpha: return mini
            else: beta = min(beta,mini[1])

        return mini    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.Expectimax(gameState, 0, self.depth)[1] #here i path the self depth then decrement it and access agents index just by agentIndex+1

    def Expectimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0: #check lose or win or depth reached 0
          return (self.evaluationFunction(gameState),'ay thing sholud stop')
        
        agentsNum = gameState.getNumAgents()
        agentIndex %=  agentsNum
        if agentIndex == agentsNum - 1: #if any ghosts turn decrement depth by one 
            depth -= 1

        if agentIndex == 0: #pacman turn so call maxval else ghost turn call minval (which is not playing optimal)
            return self.maxval(gameState, agentIndex, depth)
        else:
            return self.minval(gameState, agentIndex, depth)

    def maxval(self, gameState, agentIndex, depth):
        actions = [] #here max is playing optimal so i will return its maximum 
        for action in gameState.getLegalActions(agentIndex):
            actions.append((self.Expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action))   #append score and its corrsponding action
        return max(actions)#reurn the max is it plays optimal

        
    
    def minval(self, gameState, agentIndex, depth):
        actions = [] #here min is playing so here i will not return min as minimax but i will return average
        total = 0
        for action in gameState.getLegalActions(agentIndex):
            v = self.Expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0] 
            total += v
            actions.append((v, action))
        
        return (total / len(actions), ) #return the average as min (ghost ) is not playing optimal so i wll not return min
        

import math
import numpy as np        
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()		
    newFood = currentGameState.getFood()				
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    numFood = currentGameState.getNumFood()

    ghost_dists = [manhattanDistance(pos, newPos) for pos in currentGameState.getGhostPositions()] #get all ghost distances in arr
    food_dists = [manhattanDistance(pos, newPos) for pos in newFood.asList()]# get all food distnces in arr
    capsule_dists =[manhattanDistance(pos,newPos) for pos in currentGameState.getCapsules()]

    food_min = 0.0 if len(food_dists)==0 else np.min(food_dists)#if there is no food left food_min=0
    capsule_min = 0.0 if len(capsule_dists)==0 else np.min(capsule_dists)
    ghost_min = ghost_dists[0] if newScaredTimes[0]==0 else 1000.0 # if there is scared time make the distance between ghost and pac big enough

    capsule_min = 0.0 if len(capsule_dists)==0 else np.min(capsule_dists)
	
    return currentGameState.getScore()- 1.0/(ghost_min+1) + 1.0/(food_min+1) + 1.0/(capsule_min+1) +100
	

# Abbreviation
better = betterEvaluationFunction