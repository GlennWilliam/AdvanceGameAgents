from A2377948_KInARow import OurAgent
from game_types import TTT

agent = OurAgent()
agent.prepare(TTT, 'X', 'Opponent', apis_ok=True)

state = TTT.initial_state
remark = "Give me an AI response"
print(agent.make_move(state, remark))

agent.prepare(TTT, 'X', 'Opponent', apis_ok=False)
print(agent.make_move(state, remark))

