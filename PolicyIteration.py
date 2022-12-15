

gamma: float = 0.99
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class State:
	loc: tuple
	reward: int
	actions: dict = {}


	def __init__(self, loc: tuple) -> None:
		self.loc = loc
		self.build_actions()
	
	def __str__(self) -> str:
		return str(self.reward)

	def build_actions(self) -> None:
		self.actions = {}
		if self.loc[1] == 0 and (self.loc[0] == 0 or self.loc[0] == 2):
			return
		
		self.actions[LEFT] = self.action_results(LEFT)
		self.actions[RIGHT] = self.action_results(RIGHT)
		self.actions[UP] = self.action_results(UP)
		self.actions[DOWN] = self.action_results(DOWN)
	

	def action_results(self, a) -> list:
		leftLoc = (max(self.loc[0]-1, 0), self.loc[1])
		rightLoc = (min(self.loc[0]+1, 2), self.loc[1])
		upLoc = (self.loc[0], max(self.loc[1]-1, 0))
		downLoc = (self.loc[0], min(self.loc[1]+1, 2))
		result: list = []
		if a == LEFT:
			result.append(ActionResult(0.8, leftLoc))
			result.append(ActionResult(0.1, upLoc))
			result.append(ActionResult(0.1, downLoc))
		elif a == RIGHT:
			result.append(ActionResult(0.8, rightLoc))
			result.append(ActionResult(0.1, upLoc))
			result.append(ActionResult(0.1, downLoc))
		elif a == UP:
			result.append(ActionResult(0.8, upLoc))
			result.append(ActionResult(0.1, leftLoc))
			result.append(ActionResult(0.1, rightLoc))
		elif a == DOWN:
			result.append(ActionResult(0.8, downLoc))
			result.append(ActionResult(0.1, leftLoc))
			result.append(ActionResult(0.1, rightLoc))
		return result

	def get_state_utility(self, action: int, board: list, statesUtils: list) -> float:
		if len(self.actions) == 0: return 0
		utility: float = 0
		i: ActionResult = None
		results: list = self.actions[action]
		for i in results: #Checking each possible result of an action
			nextLoc: tuple = i.resultLoc
			nextStatePrevUtil = statesUtils[nextLoc[0] + 3*nextLoc[1]]
			utility += i.chance * (board[nextLoc[0]][nextLoc[1]].reward + gamma*nextStatePrevUtil)
		return utility


class ActionResult:
	resultLoc: tuple
	chance: float

	def __init__(self, chance: float, resultLoc: tuple) -> None:
		self.resultLoc = resultLoc
		self.chance = chance



def policyIteration(r: int) -> tuple:
	statesGrid: list = []

	for x in range(3):
		col: list = []
		for y in range(3):
			newSquare: State = State((x,y))
			newSquare.reward = -1
			
			col.append(newSquare)
		statesGrid.append(col)
	
	statesGrid[0][0].reward = r
	statesGrid[2][0].reward = 10
	previousIterationPolicies: list = [
		None, LEFT, None,
		UP, LEFT, DOWN,
		UP, LEFT, LEFT];
	previousIterationUtilities = converge_policy_utilities(statesGrid, previousIterationPolicies)
	converged: bool = False
	while not converged: #Policy iteration loop
		newPolicies: list = [None] * 9
		newUtilities: list = [None] * 9
		
		for x in range(3):
			for y in range(3): #States loop
				state: State = statesGrid[x][y]
				if len(state.actions) == 0:
					newPolicies[x + 3*y] = None
					newUtilities[x + 3*y] = 0
					continue
				
				utilities: dict = {}
				for action in state.actions: #Going through each action in a state
					utilities[action] = state.get_state_utility(action, statesGrid, previousIterationUtilities)
				newPolicies[x + 3*y] = max(utilities, key=utilities.get)
				newUtilities[x + 3*y] = max(utilities.values())
		
		converged = True
		for i in range(9):
			oldPolicy = previousIterationPolicies[i]
			newPolicy = newPolicies[i]
			if oldPolicy != newPolicy: 
				converged = False
				break
		previousIterationPolicies = newPolicies
		if not converged:
			previousIterationUtilities = converge_policy_utilities(statesGrid, previousIterationPolicies, newUtilities)
		else: previousIterationUtilities = newUtilities
	
	return previousIterationPolicies, previousIterationUtilities


def converge_policy_utilities(board: list, policies: list, initialUtils: list = [0]*9, converganceThreshold: float = 0.001) -> list:
	currentError = 1
	while currentError > converganceThreshold:
		nextUtils: list = [0] * 9
		for x in range(3):
			for y in range(3):#States loop
				state: State = board[x][y]
				action: int  = policies[x + 3*y]
				nextUtils[x + 3*y] = state.get_state_utility(action, board, initialUtils)
		maxNew: float  = max(nextUtils)
		maxOld: float = max(initialUtils)
		maxNew = maxNew if nextUtils.count(maxNew) else maxNew*-1
		maxOld = maxOld if nextUtils.count(maxOld) else maxOld*-1
		
		currentError = abs(maxNew - maxOld)
		initialUtils = nextUtils
	return initialUtils

