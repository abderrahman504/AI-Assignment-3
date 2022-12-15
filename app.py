from PolicyIteration import *



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

