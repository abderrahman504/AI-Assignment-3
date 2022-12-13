from Gauss_Jordan import Gauss_Jordan


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
		

class ActionResult:
	resultLoc: tuple
	chance: float

	def __init__(self, chance: float, resultLoc: tuple) -> None:
		self.resultLoc = resultLoc
		self.chance = chance





def policyIteration(r: int) -> tuple:
	gamma: float = 0.99
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
	#Solve the system of equations and store result in previousIterationUtilitiesp
	systemA = [
		[1         , 0          , 0         , 0          , 0         , 0          , 0          , 0          , 0],#1
		[0         , 1-0.1*gamma, -0.8*gamma, 0          , -0.1*gamma, 0          , 0          , 0          , 0],#2
		[0         , 0          , 1         , 0          , 0         , 0          , 0          , 0          , 0],#3
		[-0.8*gamma, 0          , 0         , 1-0.1*gamma, -0.1*gamma, 0          , 0          , 0          , 0],#4
		[0         , -0.8*gamma , 0         , -0.1*gamma , 1         , -0.1*gamma , 0          , 0          , 0],#5
		[0         , 0          , -0.8*gamma, 0          , -0.1*gamma, 1-0.1*gamma, 0          , 0          , 0],#6
		[0         , 0          , 0         , -0.8*gamma , 0         , 0          , 1-0.1*gamma, -0.1*gamma , 0],#7
		[0         , 0          , 0         , 0          , -0.8*gamma, 0          , -0.1*gamma , 1          , -0.1*gamma],#8
		[0         , 0          , 0         , 0          , 0         , -0.8*gamma , 0          , -0.1*gamma , 1-0.1*gamma]#9
	];
	systemB = [
		0,
		7.8,
		0,
		0.8*r-0.2,
		-1,
		7.8,
		-1,
		-1,
		-1];
	previousIterationUtilities = Gauss_Jordan(systemA, systemB, 9, 10)
	previousIterationPolicies: list = [
		None, RIGHT, None,
		UP, UP, UP,
		UP, UP, UP];
	converged: bool = False
	while not converged: #Policy iteration loop
		newPolicies: list = []
		newUtilities: list = []
		for i in range(9):
			newPolicies.append(None)
			newUtilities.append(None)
		
		for x in range(3):
			for y in range(3): #States loop
				state: State = statesGrid[x][y]
				if len(state.actions) == 0:
					newPolicies[x + 3*y] = None
					newUtilities[x + 3*y] = 0
					continue
				
				utilities: dict = {}
				for action in state.actions: #Going through each action in a state
					results: list = state.actions[action]
					utility: float = 0
					i: ActionResult = None
					for i in results: #Checking each possible result of an action
						nextLoc: tuple = i.resultLoc
						nextStatePrevUtil = previousIterationUtilities[nextLoc[0] + 3*nextLoc[1]]
						utility += i.chance * (statesGrid[nextLoc[0]][nextLoc[1]].reward + gamma*nextStatePrevUtil)
					utilities[action] = utility
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
		previousIterationUtilities = newUtilities
	
	return previousIterationPolicies, previousIterationUtilities



def print_policy(policies: list, utilities: list) -> None:
	print("Optimal Policy:")
	for y in range(3):
		row: list = []
		for x in range(3):
			policy: int = policies[x + 3*y]
			util: float = utilities[x + 3*y]
			if policy == LEFT:
				string = "Left:"
			elif policy == RIGHT:
				string = "Right:"
			elif policy == UP:
				string = "Up:"
			elif policy == DOWN:
				string = "Down:"
			else:
				string = "None:"
			string = string + str(util)
			row.append(string)
		print(row)


testR = 100
print("r = " + str(testR))
optimalPolicies, optimalUtilities = policyIteration(testR)
print_policy(optimalPolicies, optimalUtilities)
print()


testR = 3
print("r = " + str(testR))
optimalPolicies, optimalUtilities = policyIteration(testR)
print_policy(optimalPolicies, optimalUtilities)
print()


testR = 0
print("r = " + str(testR))
optimalPolicies, optimalUtilities = policyIteration(testR)
print_policy(optimalPolicies, optimalUtilities)
print()


testR = -3
print("r = " + str(testR))
optimalPolicies, optimalUtilities = policyIteration(testR)
print_policy(optimalPolicies, optimalUtilities)
print()
		


		
