# Implementation of an 8-puzzle solution using Breadth First Search, Depth First Search and A*

import sys
import time
from collections import deque as dq
from collections import namedtuple as nt
# from queue import PriorityQueue as pq
import heapq

# starting the timer
start_time = time.time()

# Size of the puzzle, integer
PUZZLE_SIZE = 8


class Node(nt("Node", ["board", "parent", "pos0", "direction"])):

    """
    A class to represent the nodes examined

    board: is the current instance of the board
    parent: is the parent node
    pos0: current pos0 of the node

    """
    def __new__(cls, board, parent=None, pos0=None, direction=None):
        if pos0 is None:
            pos0 = board.index(0)
        return super().__new__(cls, board, parent, pos0, direction)

    def __init__(self, board, parent, pos0, direction):
        super().__init__()
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def __eq__(self, node):
        return self.board == node.board

    def __hash__(self):
        return hash(self.board)

    def moves(self, method):

        # check which method is used
        if method == "bfs" or method == "ast":
            # check if "Up" is a valid move
            if self.pos0 >= 3:
                yield self.create(self.pos0 - 3, 1)
            # check if "Down" is a valid move
            if (self.pos0 + 3) <= PUZZLE_SIZE:
                yield self.create(self.pos0 + 3, 2)
            # check if "Left" is a valid move
            if self.pos0 % 3:
                yield self.create(self.pos0 - 1, 3)
            # check if "Right" is a valid move
            if (self.pos0 + 1) % 3:
                yield self.create(self.pos0 + 1, 4)

        elif method == "dfs":
            # check if "Right" is a valid move
            if (self.pos0 + 1) % 3:
                yield self.create(self.pos0 + 1, 1)
            # check if "Left" is a valid move
            if self.pos0 % 3:
                yield self.create(self.pos0 - 1, 2)
            # check if "Down" is a valid move
            if (self.pos0 + 3) <= PUZZLE_SIZE:
                yield self.create(self.pos0 + 3, 3)
            # check if "Up" is a valid move
            if self.pos0 >= 3:
                yield self.create(self.pos0 - 3, 4)

    def create(self, loc, direction):

        board = list(self.board)

        x, y = self.pos0, loc

        board[x], board[y] = board[y], board[x]

        return Node(tuple(board), parent=self, pos0=loc, direction=direction)

    def path(self):

        node = self

        path = []

        while node.parent:

            parent_loc = node.parent.board.index(0)
            child_loc = node.pos0

            distance = child_loc - parent_loc

            if distance == 3:
                path.append("Down")
            elif distance == -3:
                path.append("Up")
            elif distance == 1:
                path.append("Right")
            elif distance == -1:
                path.append("Left")

            node = node.parent

        return list(reversed(path))

    def manhattan(self):

        cost = 0

        for pos in range(1, PUZZLE_SIZE+1):

            current_index = self.index_2d(self.board.index(pos), 3)

            goal_index = self.index_2d(pos, 3)

            cost += abs(goal_index[0] - current_index[0])+abs(goal_index[1]-current_index[1])

        return cost

    @staticmethod
    def index_2d(pos, width):
        """
            finds the the index of a flattened index position in a 2-Dimensional "width" wide space
            """
        x = 0
        y = 0

        if pos < width:
            x = 1
        elif width <= pos < 2 * width:
            x = 2
        elif pos >= 2 * width:
            x = 3
        if not pos % width:
            y = 1
        elif (pos + 1) % width:
            y = 2
        elif pos % width:
            y = 3

        return x, y


class Board:
    """
    A class to represent the board of the puzzle
    """
    def __init__(self, current_board):
        # define the goal board, serves as a criterion to stop execution
        self.goal_board = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        # check that the data inserted are of type list
        # assert current_board is , "Board must be of type list!"
        # check that the length of the list inserted complies with the size of the puzzle
        assert len(current_board) == PUZZLE_SIZE + 1, "The size of the board must be equal to" + str(PUZZLE_SIZE)
        self._current = tuple(current_board)
        self.initial = tuple(current_board)

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, node):
        """
        Function that returns a copy of the current board
        :param node: a node of class Node to create the board
        :return: updates the current board to the new board
        """

        prev_pos, next_pos = node.parent.pos0, node.pos0

        board = list(node.parent.board)

        board[prev_pos], board[next_pos] = board[next_pos], board[prev_pos]

        self._current = tuple(board)

    def check(self):
        if self.current == self.goal_board:
            return True
        else:
            return False


class Frontier(dq):
    """
    A class to represent the frontier space

    """
    def __init__(self):
        dq.__init__(self)
        self.ast = []   # pq()
        self._aa = 0

    def enque(self, node, method):

        """
        Function to add to a frontier list the next elements depending
        on the method selected

        Input: method: string, accepts one of three methods ("bfs", "dfs", "ast")

        result:
        """

        if method == "bfs":

            self.append(node)

        elif method == "dfs":

            self.append(node)

        elif method == "ast":
            self._aa += 1
            heapq.heappush(self.ast, (node.manhattan()+node.depth, self._aa, node))    # node.direction, self._aa
        else:
            raise ValueError("Invalid search method provided")

    def deque(self, method):

        """
        Function to remove from a frontier list the next elements depending
        on the method selected

        Input: method: string, accepts one of three methods ("bfs", "dfs", "ast")

        result:
        """

        if method == "bfs":
            return self.popleft()

        elif method == "dfs":
            return self.pop()

        elif method == "ast":
            return heapq.heappop(self.ast)[2]

        else:
            raise ValueError("Invalid search method provided")


class Solver:

    """
    TBD
    """

    def __init__(self, method, board):
        self._method = method
        self.board = tuple(board)
        self.path = []
        self.cost = 0
        self.nodes_expanded = 0
        self.search_depth = 0
        self.max_search_depth = 0

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, search_method):
        self._method = search_method

    def solve(self):

        """
        A function that performs breadth first search on a given board

        :return: solution if any
        """

        # Initiate a new board instance to store the puzzle
        board = Board(self.board)
        # Copy the method chosen to a local variable
        method = self.method

        # Initiate the first node of the puzzle which lies on the initial zero position
        node = Node(board.current, None, board.current.index(0), None)
        # Initiate a new frontier instance to store the nodes currently in the frontier
        frontier = Frontier()
        # Enque the first set of nodes to be examined in the frontier
        frontier.enque(node, method)
        # initiate a count for the number of nodes expanded
        count = 0
        # initiate a set to store all the nodes explored
        explored = set()
        # add the starting node to the set
        explored.add(node)
        # iterate while frontier is not empty
        while frontier or frontier.ast:
            # remove the node to be examined from the frontier
            node = frontier.deque(method)
            if node.parent is not None:
                # set the current board to the one defined by the node
                board.current = node
            # check if the board is the goal board
            if board.check():
                self.nodes_expanded = count
                self.path = node.path()
                self.cost = len(self.path)
                self.search_depth = len(self.path)
                # if true return the board instance
                output(self)
                return board
            # increase the count of nodes expanded by one
            count += 1

            # iterate over possible moves for the current node
            for move in node.moves(method):
                # check if the move has been explored or added to the frontier
                if move not in explored:
                    # if not enque the new node in the frontier set
                    frontier.enque(move, method)
                    # add the move to the explored set
                    explored.add(move)
                    self.max_search_depth = check_max_depth(move, self.max_search_depth)


def check_max_depth(node, depth):
    # check if a new max depth is reached
    if node.depth > depth:
        depth = node.depth
    return depth


# creates an output file containing the results
def output(solution):
    import psutil

    f = open('output.txt', 'w')
    f.write('path_to_goal:' + str(solution.path) + '\n')
    f.write('cost_of_path:' + str(solution.cost) + '\n')
    f.write('nodes_expanded:' + str(solution.nodes_expanded) + '\n')
    f.write('search_depth:' + str(solution.search_depth) + '\n')
    f.write('max_search_depth:' + str(solution.max_search_depth) + '\n')
    f.write('running_time:' + str(time.time()-start_time) + '\n')
    f.write('max_ram_usage:' + str(psutil.Process().memory_info().rss) + '\n')
    f.close()


if __name__ == '__main__':

    # converting the string input list to integers
    # input_board = [int(x) for x in sys.argv[2].split(',')]
    input_board = [2, 4, 1, 3, 6, 5, 0, 7, 8]
    # create a Solver() class object to pass the inputs
    #puzzle = Solver(sys.argv[1], input_board)
    puzzle = Solver("ast", input_board)
    puzzle.solve()
