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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0.2*successorGameState.getScore()

        successorGhostPos = successorGameState.getGhostPositions()
        newFoodList = newFood.asList()
        capsulePos = currentGameState.getCapsules()

        if newFoodList:
            minDist = min([manhattanDistance(newPos, food) for food in newFoodList])
            score += 0.2/minDist
        if successorGhostPos:
            minDist = min([manhattanDistance(newPos, ghost) for ghost in successorGhostPos])
            if minDist < 3 and max(newScaredTimes) == 0:
                score += 0.6*minDist
        if capsulePos and newPos in capsulePos:
            score += 500
        return score
def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        potentialMoves = gameState.getLegalActions(0)
        return max(potentialMoves, key = lambda action: self.helperMethod(1, 0, gameState.generateSuccessor(0, action)))

    def helperMethod(self, currAgent, currDepth, currState):
        potentialMoves = currState.getLegalActions(currAgent)
        if currDepth == self.depth or currState.isLose() or currState.isWin():
            return self.evaluationFunction(currState)
        elif currAgent == 0:
            newStates = [currState.generateSuccessor(currAgent, action) for action in potentialMoves]
            return max([self.helperMethod(currAgent + 1, currDepth, newState) for newState in newStates])
        else:
            newStates = [currState.generateSuccessor(currAgent, action) for action in potentialMoves]
            if currAgent == currState.getNumAgents() - 1:
                return min([self.helperMethod(0, currDepth + 1, newState) for newState in newStates])
            return min([self.helperMethod(currAgent + 1, currDepth, newState) for newState in newStates])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.helperMethod(0, 0, gameState, float("-inf"), float("inf"))[1]

    def helperMethod(self, currAgent, currDepth, currState, alpha, beta):
        potentialMoves = currState.getLegalActions(currAgent)
        if currDepth == self.depth or currState.isLose() or currState.isWin():
            return (self.evaluationFunction(currState), None)
        elif currAgent == 0:
            optimalAct = None
            optimalVal = float("-inf")
            for move in potentialMoves:
                currVal = self.helperMethod(currAgent + 1, currDepth, currState.generateSuccessor(currAgent, move), alpha, beta)[0]
                if optimalVal < currVal:
                    optimalVal = currVal
                    optimalAct = move
                if currVal > beta:
                    return (currVal, move)
                alpha = max(alpha, currVal)
            return (optimalVal, optimalAct)
        else:
            optimalAct = None
            optimalVal = float("inf")
            for move in potentialMoves:
                if currAgent == currState.getNumAgents() - 1:
                    currVal = self.helperMethod(0, currDepth + 1, currState.generateSuccessor(currAgent, move), alpha, beta)[0]
                else:
                    currVal = self.helperMethod(currAgent + 1, currDepth, currState.generateSuccessor(currAgent, move), alpha, beta)[0]
                if optimalVal > currVal:
                    optimalVal = currVal
                    optimalAct = move
                if currVal < alpha:
                    return (currVal, move)
                beta = min(beta, currVal)
            return (optimalVal, optimalAct)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        potentialMoves = gameState.getLegalActions(0)
        return max(potentialMoves, key = lambda move: self.helperMethod(1, 0, gameState.generateSuccessor(0, move)))

    def helperMethod(self, currAgent, currDepth, currState):
        potentialMoves = currState.getLegalActions(currAgent)
        if currDepth == self.depth or currState.isWin() or currState.isLose():
            return self.evaluationFunction(currState)
        elif currAgent == 0:
            newStates = [currState.generateSuccessor(currAgent, move) for move in potentialMoves]
            return max([self.helperMethod(currAgent + 1, currDepth, state) for state in newStates])
        else:
            newStates = [currState.generateSuccessor(currAgent, move) for move in potentialMoves]
            if currAgent == currState.getNumAgents() - 1:
                return sum([self.helperMethod(0, currDepth + 1, state) * 1 / len(newStates) for state in newStates])
            return sum([self.helperMethod(currAgent + 1, currDepth, state) * 1 / len(newStates) for state in newStates])

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    position = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    newGhostStates = currentGameState.getGhostStates()

    for ghost in newGhostStates:
        if manhattanDistance(position, ghost.getPosition()) > 1.5:
            continue
        else:
            if ghost.scaredTimer == 0:
                score = score - 46
            else:
                score = score + 23
    distance = []
    foodList = currentGameState.getFood().asList()

    if len(foodList) > 0:
        for food in foodList:
            distance.append(manhattanDistance(food, position))
        distance.sort()
        if len(distance) > 1:
            score = score - distance[1]
        else:
            score = score - distance[0]
    return score



# Abbreviation
better = betterEvaluationFunction
