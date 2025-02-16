
import numpy as np
import copy

class MCTSNode(object):

    def __init__(self, state, parent=None):
        
        self.state = state
        self.parent = parent
        self.children = []
        self.n = 0
        self.q = 0
        
        self._untried_actions = None
        
    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        return possible_moves[np.random.randint(len(possible_moves))]

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = type(self)(
            next_state, parent=self
        )
       
        self.children.append(child_node)
       
        return child_node

    def is_terminal_node(self):
        return self.state.is_terminal()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.rollout_result()

    def backpropagate(self, result):
        self.n += 1
        self.q = result
        current = self.parent
        while current is not None:
            current.n += 1
            current.q = np.mean([child.q for child in current.children])
            current = current.parent



