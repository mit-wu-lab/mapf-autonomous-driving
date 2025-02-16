import time


class MCTS(object):

    def __init__(self, node):
       
        self.root = node

    def best_action(self, num_simulations=None, total_simulation_time=None):
    

        if num_simulations is None :
            assert(total_simulation_time is not None)
            end_time = time.time() + total_simulation_time
            while True:
                node = self._tree_policy()
                reward = node.rollout()
                node.backpropagate(reward)
                if time.time() > end_time:
                    break
        else :
            for i in range(0, num_simulations):            
                node = self._tree_policy()
                reward = node.rollout()
                node.backpropagate(reward)
        
        return self.root.best_child(c_param=0.1)
        

    def _tree_policy(self):
       
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                
                return current_node.expand()
            else:
                current_node = current_node.best_child()
                
        return current_node
