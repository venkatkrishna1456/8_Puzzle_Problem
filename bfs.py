import numpy as np
from abc import ABC, abstractmethod
from collections import deque

# Finding path
class Solver(ABC):
    solution = None
    frontier = None
    max_depth = 0
    explored_nodes = set()
    initial_state = None

    def __init__(self, initial_state):
        self.initial_state = initial_state

    # Get path and save
    def ancestral_chain(self):
        current = self.solution
        chain = [current]
        while current.parent is not None:
            chain.append(current.parent)
            current = current.parent
        return chain

    # print path
    @property
    def path(self):
        path = " ".join(node.operator for node in self.ancestral_chain()[-2::-1])
        for node in self.ancestral_chain()[-2::-1]:
            print("Current board layout:")
            print(node)
        return path

    @abstractmethod
    def solve(self):
        pass

    def set_solution(self, board):
        self.solution = board

# BFS main class
class BFS(Solver):
    def __init__(self, initial_state, goal):
        super(BFS, self).__init__(initial_state)
        self.frontier = deque()
        self.goal = goal

    # Check nearest neighbor
    def solve(self):
        time_count = 0
        count = 0
        self.frontier.append(self.initial_state)
        while self.frontier:
            board = self.frontier.popleft()
            self.explored_nodes.add(tuple(board.state))
            if board.goal_test(self.goal):
                self.set_solution(board)
                break
            for neighbor in board.neighbors():
                time_count += 1
                if tuple(neighbor.state) not in self.explored_nodes:
                    count += 1
                    self.frontier.append(neighbor)
                    self.explored_nodes.add(tuple(neighbor.state))

# Change board
class Board:
    parent = None
    state = None
    operator = None
    depth = 0
    zero = None
    cost = 0
    result_list = []

    def __init__(self, state, parent=None, operator=None, depth=0):
        self.parent = parent
        self.state = np.array(state)
        self.operator = operator
        self.depth = depth
        self.zero = self.find_0()
        self.cost = self.depth + self.manhattan()

    # check distance
    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        else:
            op_pr = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
            return op_pr[self.operator] < op_pr[other.operator]

    # set board
    def __str__(self):
        self.result_list.append(self.operator.split()[1][0])
        return str(self.state[:3]) + '\n' + str(self.state[3:6]) + '\n' + str(self.state[6:]) + str(self.operator) + '\n'

    # check current state is goal state
    def goal_test(self, goal):
        if np.array_equal(self.state, goal):
            return True
        else:
            return False

    # find index of 0
    def find_0(self):
        for i in range(9):
            if self.state[i] == 0:
                return i

    # calculate manhattan distance
    def manhattan(self):
        state = self.index(self.state)
        goal = self.index(np.arange(9))
        return sum((abs(state // 3 - goal // 3) + abs(state % 3 - goal % 3))[1:])

    @staticmethod
    def index(state):
        index = np.array(range(9))
        for x, y in enumerate(state):
            index[y] = x
        return index

    # update board
    def swap(self, i, j):
        new_state = np.array(self.state)
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    # move up
    def up(self):
        if self.zero > 2:
            return Board(self.swap(self.zero, self.zero - 3), self, f'\nmove UP', self.depth + 1)
        else:
            return None

    # move down
    def down(self):
        if self.zero < 6:
            return Board(self.swap(self.zero, self.zero + 3), self, f'\nmove DOWN', self.depth + 1)
        else:
            return None

    # move left
    def left(self):
        if self.zero % 3 != 0:
            return Board(self.swap(self.zero, self.zero - 1), self, f'\nmove LEFT', self.depth + 1)
        else:
            return None

    # move right
    def right(self):
        if (self.zero + 1) % 3 != 0:
            return Board(self.swap(self.zero, self.zero + 1), self, f'\nmove RIGHT', self.depth + 1)
        else:
            return None

    def neighbors(self):
        neighbors = [self.up(), self.down(), self.left(), self.right()]
        return list(filter(None, neighbors))

    __repr__ = __str__

# main function, using bfs, solve 8 puzzle
def main():
    start = np.array([1,3,4,8,0,5,7,2,6])
    goal = np.array([1,2,3,8,0,4,7,6,5])

    board = Board(start)
    bfs = BFS(board, goal)
    result = bfs.solve()
    print("Start")
    print(start.reshape(3,3))
    print()
    bfs.path
    print("Done.")
    print(goal.reshape(3,3))
    print(f"we are finished, with path ({' '.join(t for t in board.result_list)})")


if __name__ == "__main__":
    main()
